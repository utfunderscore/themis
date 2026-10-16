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
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.RejectedExecutionException;
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
        moderator = Moderator.load(bundle, Scheduling.batched(8));
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
        List<Moderation> batched = moderator.moderateAll(texts);
        for (int i = 0; i < texts.size(); i++) {
            float[] row = logits.get(i);
            Moderation expected = Moderation.of(
                    Moderator.softmax(row, 0, Category.values().length),
                    Moderator.softmax(row, Category.values().length, Action.values().length));
            assertModeration(expected, batched.get(i));
            assertModeration(expected, moderator.moderate(texts.get(i)));
        }
    }

    @Test
    void concurrentSubmissionsMatchSequentialResults() {
        List<String> texts = List.of("hello", "you are an idiot", "gg", "", "Hello 👋 café 世界", "nice base");
        List<Moderation> expected = texts.stream().map(moderator::moderate).toList();
        List<CompletableFuture<Moderation>> futures = new ArrayList<>();
        for (int round = 0; round < 5; round++) {
            for (String text : texts) {
                futures.add(CompletableFuture.supplyAsync(() -> moderator.moderate(text)));
            }
        }
        for (int i = 0; i < futures.size(); i++) {
            assertModeration(expected.get(i % texts.size()), futures.get(i).join());
        }
    }

    @Test
    void blockingFromInferenceThreadFailsInsteadOfDeadlocking() {
        List<String> slow = Collections.nCopies(8, "hello ".repeat(300));
        CompletableFuture<Moderation> nested = moderator.submitAll(slow)
                .thenApply(ignored -> moderator.moderate("hello"));
        CompletionException failure = assertThrows(CompletionException.class, nested::join);
        assertInstanceOf(IllegalStateException.class, failure.getCause());
    }

    @Test
    void probabilitiesFormDistributions() {
        Moderation result = moderator.moderate("hello world");
        assertEquals(1.0, result.categoryProbabilities().values().stream().mapToDouble(d -> d).sum(), 1e-9);
        assertEquals(1.0, result.actionProbabilities().values().stream().mapToDouble(d -> d).sum(), 1e-9);
    }

    @Test
    void rejectsIncompleteBundle(@TempDir Path directory) throws IOException {
        Files.copy(bundle.resolve("tokenizer_config.json"), directory.resolve("tokenizer_config.json"));
        assertThrows(InferenceException.class, () -> Moderator.load(directory));
        Files.writeString(directory.resolve("inference_config.json"),
                Files.readString(bundle.resolve("inference_config.json")).replace("\"slur\"", "\"other\""));
        InferenceException failure = assertThrows(InferenceException.class, () -> Moderator.load(directory));
        assertTrue(failure.getMessage().contains("labels"), failure.getMessage());
    }

    @Test
    void closedModeratorRejectsRequests() {
        Moderator closed = Moderator.load(bundle);
        closed.close();
        assertThrows(RejectedExecutionException.class, () -> closed.moderate("hello"));
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
