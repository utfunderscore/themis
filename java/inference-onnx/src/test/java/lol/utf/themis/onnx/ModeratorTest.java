package lol.utf.themis.onnx;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.ReentrantLock;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * End-to-end checks against an exported bundle. Run with -Pthemis.onnx.bundle=DIR.
 *
 * <p>tokenizer_examples.json holds input_ids produced by the bundle's slow Python
 * tokenizer ({@code tokenizer(text, truncation=True, max_length=256)}) for edge
 * cases and seeded random Unicode strings.
 */
class ModeratorTest {

    private static final double TOLERANCE = 1e-4;

    private static Path bundle;
    private static Moderator moderator;
    private static DebertaTokenizer tokenizer;

    @BeforeAll
    static void load() throws IOException {
        String property = System.getProperty("themis.onnx.bundle");
        assumeTrue(property != null && !property.isBlank(), "themis.onnx.bundle not set");
        bundle = Path.of(property);
        tokenizer = ModelBundle.read(bundle).tokenizer();
        moderator = Moderator.load(bundle, 8).join();
    }

    @AfterAll
    static void close() {
        if (moderator != null) {
            moderator.close();
        }
    }

    @Test
    void tokenizerMatchesPython() throws IOException {
        JsonArray examples;
        try (InputStream stream = ModeratorTest.class.getResourceAsStream("/tokenizer_examples.json")) {
            examples = JsonParser.parseString(new String(stream.readAllBytes(), StandardCharsets.UTF_8))
                    .getAsJsonArray();
        }
        for (JsonElement element : examples) {
            JsonObject example = element.getAsJsonObject();
            String text = example.get("text").getAsString();
            assertArrayEquals(longs(example.getAsJsonArray("input_ids")), tokenizer.encode(text).inputIds(),
                    () -> "text: " + text);
        }
    }

    @Test
    void moderationMatchesExportValidation() throws IOException {
        JsonArray examples = JsonParser.parseString(
                Files.readString(bundle.resolve("validation_examples.json"))).getAsJsonArray();
        List<String> texts = new ArrayList<>();
        List<float[]> logits = new ArrayList<>();
        for (JsonElement element : examples) {
            JsonObject example = element.getAsJsonObject();
            JsonArray batchTexts = example.getAsJsonArray("texts");
            for (int row = 0; row < batchTexts.size(); row++) {
                String text = batchTexts.get(row).getAsString();
                long[] padded = longs(example.getAsJsonArray("input_ids").get(row).getAsJsonArray());
                long[] ids = tokenizer.encode(text).inputIds();
                assertArrayEquals(Arrays.copyOf(padded, ids.length), ids, () -> "text: " + text);
                texts.add(text);
                logits.add(floats(example.getAsJsonArray("logits").get(row).getAsJsonArray()));
            }
        }
        List<Moderation> batched = moderator.moderateAll(texts).join();
        for (int i = 0; i < texts.size(); i++) {
            float[] row = logits.get(i);
            Moderation expected = Moderation.of(
                    Moderator.softmax(row, 0, Category.values().length),
                    Moderator.softmax(row, Category.values().length, Action.values().length));
            assertModeration(expected, batched.get(i));
            assertModeration(expected, moderator.moderate(texts.get(i)).join());
        }
    }

    @Test
    void concurrentSubmissionsMatchSequentialResults() {
        List<String> texts = List.of("hello", "you are an idiot", "gg", "", "Hello 👋 café 世界", "nice base");
        List<Moderation> expected = texts.stream().map(text -> moderator.moderate(text).join()).toList();
        List<CompletableFuture<Moderation>> futures = new ArrayList<>();
        for (int round = 0; round < 5; round++) {
            for (String text : texts) {
                futures.add(CompletableFuture.supplyAsync(() -> moderator.moderate(text)).thenCompose(f -> f));
            }
        }
        for (int i = 0; i < futures.size(); i++) {
            assertModeration(expected.get(i % texts.size()), futures.get(i).join());
        }
    }

    @Test
    void invalidArgumentsFailTheFutureInsteadOfThrowing() {
        assertFailsWith(NullPointerException.class, moderator.moderate(null));
        assertFailsWith(NullPointerException.class, moderator.moderateAll(null));
        assertFailsWith(NullPointerException.class, moderator.moderateAll(Arrays.asList("hello", null)));
        assertFailsWith(IllegalArgumentException.class, Moderator.load(bundle, 0));
        assertFailsWith(NullPointerException.class, Moderator.load(null));
    }

    @Test
    void probabilitiesFormDistributions() {
        Moderation result = moderator.moderate("hello world").join();
        assertEquals(1.0, result.categoryProbabilities().values().stream().mapToDouble(d -> d).sum(), 1e-9);
        assertEquals(1.0, result.actionProbabilities().values().stream().mapToDouble(d -> d).sum(), 1e-9);
    }

    @Test
    void rejectsIncompleteBundle(@TempDir Path directory) throws IOException {
        Files.copy(bundle.resolve("tokenizer_config.json"), directory.resolve("tokenizer_config.json"));
        assertFailsWith(InferenceException.class, Moderator.load(directory));
        Files.writeString(directory.resolve("inference_config.json"),
                Files.readString(bundle.resolve("inference_config.json")).replace("\"slur\"", "\"other\""));
        Throwable failure = assertFailsWith(InferenceException.class, Moderator.load(directory));
        assertTrue(failure.getMessage().contains("labels"), failure.getMessage());
    }

    @Test
    void closedModeratorRejectsRequests() {
        Moderator closed = Moderator.load(bundle).join();
        closed.close();
        assertFailsWith(RejectedExecutionException.class, closed.moderate("hello"));
        assertFailsWith(RejectedExecutionException.class, closed.moderateAll(List.of("hello")));
    }

    @Test
    void closingFromInferenceThreadDoesNotDeadlockAndFinishesQueuedRequests() throws Exception {
        try (Moderator local = Moderator.load(bundle).join()) {
            Field executorField = Moderator.class.getDeclaredField("executor");
            executorField.setAccessible(true);
            InferenceExecutor executor = (InferenceExecutor) executorField.get(local);
            Field lockField = InferenceExecutor.class.getDeclaredField("lock");
            lockField.setAccessible(true);
            ReentrantLock queueLock = (ReentrantLock) lockField.get(executor);
            CountDownLatch entered = new CountDownLatch(1);
            CountDownLatch release = new CountDownLatch(1);
            CompletableFuture<Void> callback;
            // Keep the worker from completing the request before the callback is attached.
            queueLock.lock();
            try {
                callback = local.moderate("hello").thenAccept(ignored -> {
                    assertTrue(executor.isWorkerThread());
                    entered.countDown();
                    try {
                        assertTrue(release.await(30, TimeUnit.SECONDS), "callback was not released");
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        throw new AssertionError(e);
                    }
                    local.close();
                });
            } finally {
                queueLock.unlock();
            }
            CompletableFuture<Moderation> queued;
            try {
                assertTrue(entered.await(30, TimeUnit.SECONDS), "callback did not start");
                queued = local.moderate("second message");
            } finally {
                release.countDown();
            }
            callback.get(30, TimeUnit.SECONDS);
            assertTrue(queued.get(30, TimeUnit.SECONDS).probability(Category.NONE) >= 0);
            local.closeAsync().get(30, TimeUnit.SECONDS);
            assertFailsWith(RejectedExecutionException.class, local.moderate("after close"));
        }
    }

    private static Throwable assertFailsWith(Class<? extends Throwable> type, CompletableFuture<?> future) {
        ExecutionException failure = assertThrows(ExecutionException.class, () -> future.get(30, TimeUnit.SECONDS));
        return assertInstanceOf(type, failure.getCause());
    }

    private static void assertModeration(Moderation expected, Moderation actual) {
        assertEquals(expected.category(), actual.category());
        assertEquals(expected.action(), actual.action());
        for (Category category : Category.values()) {
            assertEquals(expected.probability(category), actual.probability(category), TOLERANCE, category.label());
        }
        for (Action action : Action.values()) {
            assertEquals(expected.probability(action), actual.probability(action), TOLERANCE, action.label());
        }
    }

    private static long[] longs(JsonArray array) {
        long[] values = new long[array.size()];
        for (int i = 0; i < values.length; i++) {
            values[i] = array.get(i).getAsLong();
        }
        return values;
    }

    private static float[] floats(JsonArray array) {
        float[] values = new float[array.size()];
        for (int i = 0; i < values.length; i++) {
            values[i] = array.get(i).getAsFloat();
        }
        return values;
    }
}
