package lol.utf.themis.onnx;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** Compares against the exporter's validation_examples.json. Run with -Pthemis.onnx.bundle=DIR. */
class OnnxModelTest {

    private static final float TOLERANCE = 1e-3f;

    private record Example(List<ModelInput> inputs, float[][] logits) {
    }

    private static OnnxModel model;
    private static List<Example> examples;

    @BeforeAll
    static void load() throws IOException {
        String bundle = System.getProperty("themis.onnx.bundle");
        assumeTrue(bundle != null && !bundle.isBlank(), "themis.onnx.bundle not set");
        Path directory = Path.of(bundle);
        examples = parseExamples(directory.resolve("validation_examples.json"));
        model = OnnxModel.load(directory.resolve("model.onnx"));
    }

    @AfterAll
    static void close() {
        if (model != null) {
            model.close();
        }
    }

    @Test
    void paddedBatchesMatchPython() {
        for (Example example : examples) {
            assertLogits(example.logits(), model.run(example.inputs()));
        }
    }

    @Test
    void concurrentRunsMatchPython() throws Exception {
        ExecutorService pool = Executors.newFixedThreadPool(4);
        try {
            List<Future<?>> tasks = new ArrayList<>();
            for (int i = 0; i < 8; i++) {
                Example example = examples.get(i % examples.size());
                tasks.add(pool.submit(() -> assertLogits(example.logits(), model.run(example.inputs()))));
            }
            for (Future<?> task : tasks) {
                task.get();
            }
        } finally {
            pool.shutdownNow();
        }
    }

    @Test
    void executorsMatchPython() throws Exception {
        List<ModelInput> inputs = new ArrayList<>();
        List<float[]> expected = new ArrayList<>();
        for (Example example : examples) {
            inputs.addAll(example.inputs());
            expected.addAll(List.of(example.logits()));
        }
        float[][] want = expected.toArray(float[][]::new);
        try (InferenceExecutor sequential = InferenceExecutor.create(model, 1);
             InferenceExecutor batched = InferenceExecutor.create(model, 3)) {
            List<CompletableFuture<float[]>> single = new ArrayList<>();
            for (ModelInput input : inputs) {
                single.add(sequential.submit(input));
            }
            CompletableFuture<List<float[]>> grouped = batched.submitAll(inputs);
            assertLogits(want, single.stream().map(CompletableFuture::join).toArray(float[][]::new));
            assertLogits(want, grouped.get().toArray(float[][]::new));
        }
    }

    @Test
    void reportsLoadFailure() {
        Path missing = Path.of("does-not-exist.onnx");
        assertThrows(InferenceException.class, () -> OnnxModel.load(missing));
    }

    private static void assertLogits(float[][] expected, float[][] actual) {
        assertEquals(expected.length, actual.length);
        for (int row = 0; row < expected.length; row++) {
            assertEquals(expected[row].length, actual[row].length);
            for (int column = 0; column < expected[row].length; column++) {
                assertEquals(expected[row][column], actual[row][column], TOLERANCE,
                        "row " + row + ", column " + column);
            }
        }
    }

    private static List<Example> parseExamples(Path file) throws IOException {
        List<Example> examples = new ArrayList<>();
        for (JsonElement element : JsonParser.parseString(Files.readString(file)).getAsJsonArray()) {
            JsonObject example = element.getAsJsonObject();
            JsonArray ids = example.getAsJsonArray("input_ids");
            JsonArray masks = example.getAsJsonArray("attention_mask");
            JsonArray logits = example.getAsJsonArray("logits");
            List<ModelInput> inputs = new ArrayList<>();
            float[][] expected = new float[ids.size()][];
            for (int row = 0; row < ids.size(); row++) {
                JsonArray rowMask = masks.get(row).getAsJsonArray();
                int length = 0;
                while (length < rowMask.size() && rowMask.get(length).getAsLong() == 1) {
                    length++;
                }
                long[] rowIds = new long[length];
                for (int i = 0; i < length; i++) {
                    rowIds[i] = ids.get(row).getAsJsonArray().get(i).getAsLong();
                }
                inputs.add(ModelInput.of(rowIds));
                JsonArray rowLogits = logits.get(row).getAsJsonArray();
                expected[row] = new float[rowLogits.size()];
                for (int i = 0; i < expected[row].length; i++) {
                    expected[row][i] = rowLogits.get(i).getAsFloat();
                }
            }
            examples.add(new Example(inputs, expected));
        }
        return examples;
    }
}
