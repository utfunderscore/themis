package lol.utf.themis.onnx;

import java.io.IOException;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.time.Duration;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.regex.Pattern;

/** Downloads ONNX bundles published to the Hugging Face Hub by the Themis Python uploader. */
public final class ModelDownloader {
    /** The official Themis model repository. */
    public static final String DEFAULT_REPO_ID = "utfunderscore/themis-moderation";
    /** The {@link #DEFAULT_REPO_ID} commit this library version is built against. */
    public static final String DEFAULT_REVISION = "fcc179da34ed72795035edcc053fc56d49e0d8e8";

    private static final URI HUGGING_FACE = URI.create("https://huggingface.co/");
    private static final Pattern REPO_ID = Pattern.compile("(?:[\\w.-]+/)?[\\w.-]+");
    private static final List<String> FILES = List.of(
            "model.onnx", "spm.model", "tokenizer_config.json", "inference_config.json");

    private ModelDownloader() {}

    /** Downloads the official model into {@code themis} under the working directory. */
    public static CompletableFuture<Path> download() {
        return download(DEFAULT_REPO_ID, DEFAULT_REVISION);
    }

    /** Downloads the official model into {@code cacheDirectory}. */
    public static CompletableFuture<Path> download(Path cacheDirectory) {
        return download(DEFAULT_REPO_ID, DEFAULT_REVISION, cacheDirectory);
    }

    /**
     * Downloads into {@code themis} under the working directory, the directory the
     * JVM was started from. See {@link #download(String, String, Path)}.
     */
    public static CompletableFuture<Path> download(String repoId, String revision) {
        return download(repoId, revision, Path.of("themis"));
    }

    /**
     * Downloads the four required files from the {@code onnx/} folder of a public
     * Hugging Face model repository, such as {@code "OWNER/MODEL"}, into a
     * source-specific directory under {@code cacheDirectory}. Returns the directory
     * to pass to {@link Moderator#load(Path)}.
     *
     * <p>Use a commit ID as {@code revision}. Branch and tag names are accepted, but
     * completed downloads are reused without network access or freshness checks,
     * so a cached branch never updates. Delete the cached directory to download again.
     *
     * <p>Transfers run on a background thread and stream to disk. A bundle becomes
     * visible only after all files download successfully. Concurrent downloads
     * are serialized, including across processes sharing the cache. HTTP errors,
     * empty files and I/O failures fail the future with {@link IOException}.
     * Model compatibility is checked separately by {@code Moderator.load}.
     * Private and gated repositories are not supported.
     */
    public static CompletableFuture<Path> download(String repoId, String revision, Path cacheDirectory) {
        return download(HUGGING_FACE, repoId, revision, cacheDirectory);
    }

    static CompletableFuture<Path> download(URI endpoint, String repoId, String revision, Path cacheDirectory) {
        CompletableFuture<Path> result = new CompletableFuture<>();
        try {
            Objects.requireNonNull(repoId, "repoId");
            Objects.requireNonNull(revision, "revision");
            Objects.requireNonNull(cacheDirectory, "cacheDirectory");
            if (!REPO_ID.matcher(repoId).matches() || repoId.contains("..")) {
                throw new IllegalArgumentException("repoId must look like OWNER/MODEL: " + repoId);
            }
            if (revision.isBlank()) {
                throw new IllegalArgumentException("revision must not be blank");
            }
            String encodedRevision = URLEncoder.encode(revision, StandardCharsets.UTF_8).replace("+", "%20");
            URI source = endpoint.resolve(repoId + "/resolve/" + encodedRevision + "/onnx/");
            Thread.ofVirtual().name("themis-model-download").start(() -> {
                try {
                    result.complete(fetch(source, cacheDirectory.toAbsolutePath().normalize()));
                } catch (Exception e) {
                    if (e instanceof InterruptedException) {
                        Thread.currentThread().interrupt();
                    }
                    result.completeExceptionally(e);
                }
            });
        } catch (RuntimeException e) {
            result.completeExceptionally(e);
        }
        return result;
    }

    // JVM serialization avoids OverlappingFileLockException; the file lock handles other processes.
    private static synchronized Path fetch(URI source, Path cache) throws Exception {
        String key = HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256")
                .digest(source.toASCIIString().getBytes(StandardCharsets.UTF_8)));
        Files.createDirectories(cache);
        try (FileChannel channel = FileChannel.open(cache.resolve(key + ".lock"),
                StandardOpenOption.CREATE, StandardOpenOption.WRITE);
             var lock = channel.lock()) {
            Path destination = cache.resolve(key);
            if (Files.isDirectory(destination)) {
                for (String name : FILES) {
                    requireFile(destination.resolve(name));
                }
                return destination;
            }
            Path staging = Files.createTempDirectory(cache, key + ".partial-");
            try (HttpClient client = HttpClient.newBuilder()
                    .connectTimeout(Duration.ofSeconds(30))
                    .followRedirects(HttpClient.Redirect.NORMAL).build()) {
                for (String name : FILES) {
                    Path file = staging.resolve(name);
                    HttpRequest request = HttpRequest.newBuilder(source.resolve(name))
                            .timeout(Duration.ofMinutes(30)).GET().build();
                    HttpResponse<Path> response = client.send(request, HttpResponse.BodyHandlers.ofFile(file));
                    if (response.statusCode() != 200) {
                        throw new IOException("Download failed for " + request.uri() + ": HTTP " + response.statusCode());
                    }
                    requireFile(file);
                }
                Files.move(staging, destination, java.nio.file.StandardCopyOption.ATOMIC_MOVE);
                return destination;
            } finally {
                if (Files.exists(staging)) {
                    try (var paths = Files.walk(staging)) {
                        for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) {
                            Files.deleteIfExists(path);
                        }
                    }
                }
            }
        }
    }

    private static void requireFile(Path file) throws IOException {
        if (!Files.isRegularFile(file) || Files.size(file) == 0) {
            throw new IOException("Missing or empty model file: " + file);
        }
    }
}
