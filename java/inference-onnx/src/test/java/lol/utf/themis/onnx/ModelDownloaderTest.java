package lol.utf.themis.onnx;

import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import static org.junit.jupiter.api.Assertions.*;

class ModelDownloaderTest {
    private static final String REPO = "owner/model";
    private static final String COMMIT = "0123abcd";

    @TempDir Path cache;
    private HttpServer server;
    private URI endpoint;
    private final AtomicInteger requests = new AtomicInteger();
    private volatile int modelStatus = 200;
    private volatile boolean emptyModel;

    @BeforeEach
    void startServer() throws IOException {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/" + REPO + "/resolve/", exchange -> {
            requests.incrementAndGet();
            String path = exchange.getRequestURI().getRawPath();
            boolean model = path.endsWith("model.onnx");
            // Hugging Face serves LFS files such as model.onnx through a CDN redirect.
            if (model && modelStatus == 200 && !emptyModel) {
                exchange.getResponseHeaders().add("Location", "/cdn" + path);
                exchange.sendResponseHeaders(302, -1);
                exchange.close();
                return;
            }
            byte[] bytes = path.getBytes(StandardCharsets.UTF_8);
            exchange.sendResponseHeaders(model ? modelStatus : 200, model && emptyModel ? -1 : bytes.length);
            try (var body = exchange.getResponseBody()) {
                if (!(model && emptyModel)) body.write(bytes);
            }
        });
        server.createContext("/cdn/", exchange -> {
            byte[] bytes = exchange.getRequestURI().getRawPath().substring(4).getBytes(StandardCharsets.UTF_8);
            exchange.sendResponseHeaders(200, bytes.length);
            try (var body = exchange.getResponseBody()) {
                body.write(bytes);
            }
        });
        server.start();
        endpoint = URI.create("http://127.0.0.1:" + server.getAddress().getPort() + "/");
    }

    @AfterEach
    void stopServer() {
        server.stop(0);
    }

    @Test
    void downloadsOnnxFolderAndReusesCacheOffline() throws IOException {
        Path bundle = ModelDownloader.download(endpoint, REPO, COMMIT, cache).join();
        for (String name : new String[]{"model.onnx", "spm.model", "tokenizer_config.json", "inference_config.json"}) {
            assertEquals("/" + REPO + "/resolve/" + COMMIT + "/onnx/" + name, Files.readString(bundle.resolve(name)));
        }
        assertEquals(4, requests.get());
        server.stop(0);
        assertEquals(bundle, ModelDownloader.download(endpoint, REPO, COMMIT, cache).join());
        assertEquals(4, requests.get());
    }

    @Test
    void concurrentCallsShareOneDownload() {
        var first = ModelDownloader.download(endpoint, REPO, COMMIT, cache);
        var second = ModelDownloader.download(endpoint, REPO, COMMIT, cache);
        CompletableFuture.allOf(first, second).join();
        assertEquals(first.join(), second.join());
        assertEquals(4, requests.get());
    }

    @Test
    void failureCleansUpAndCanBeRetried() throws IOException {
        modelStatus = 404;
        CompletionException failure = assertThrows(CompletionException.class,
                () -> ModelDownloader.download(endpoint, REPO, COMMIT, cache).join());
        assertInstanceOf(IOException.class, failure.getCause());
        assertTrue(failure.getCause().getMessage().contains("404"));
        try (var entries = Files.list(cache)) {
            assertTrue(entries.noneMatch(Files::isDirectory));
        }
        modelStatus = 200;
        assertTrue(Files.isDirectory(ModelDownloader.download(endpoint, REPO, COMMIT, cache).join()));
    }

    @Test
    void rejectsEmptyResponse() {
        emptyModel = true;
        assertInstanceOf(IOException.class, assertThrows(CompletionException.class,
                () -> ModelDownloader.download(endpoint, REPO, COMMIT, cache).join()).getCause());
    }

    @Test
    void separatesRevisionsAndEncodesThem() throws IOException {
        Path first = ModelDownloader.download(endpoint, REPO, COMMIT, cache).join();
        Path second = ModelDownloader.download(endpoint, REPO, "refs/pr/1", cache).join();
        assertNotEquals(first, second);
        assertEquals("/" + REPO + "/resolve/refs%2Fpr%2F1/onnx/spm.model", Files.readString(second.resolve("spm.model")));
        assertEquals(8, requests.get());
    }

    @Test
    void invalidArgumentsFailThroughFuture() {
        for (String repo : new String[]{"", "a/b/c", "../model", "owner/model?x=1", "https://huggingface.co/owner/model"}) {
            assertInstanceOf(IllegalArgumentException.class, assertThrows(CompletionException.class,
                    () -> ModelDownloader.download(repo, COMMIT, cache).join()).getCause(), repo);
        }
        assertInstanceOf(IllegalArgumentException.class, assertThrows(CompletionException.class,
                () -> ModelDownloader.download(REPO, " ", cache).join()).getCause());
        assertInstanceOf(NullPointerException.class, assertThrows(CompletionException.class,
                () -> ModelDownloader.download(REPO, COMMIT, null).join()).getCause());
    }
}
