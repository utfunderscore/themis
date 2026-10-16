package lol.utf.themis.onnx;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * An exported model taking int64 {@code input_ids} and {@code attention_mask}
 * of shape {@code [batch, sequence]} and returning float32 {@code logits} of
 * shape {@code [batch, classes]}.
 *
 * <p>{@link #run} may be called concurrently; ONNX Runtime sessions support
 * concurrent runs. Inputs are right-padded to the longest sequence in the batch.
 * {@link #close} waits for in-flight runs to finish.
 */
final class OnnxModel implements InferenceModel, AutoCloseable {

    static final String INPUT_IDS = "input_ids";
    static final String ATTENTION_MASK = "attention_mask";
    static final String LOGITS = "logits";

    private final OrtEnvironment environment;
    private final OrtSession session;
    private final long padTokenId;
    private final ReadWriteLock lifecycle = new ReentrantReadWriteLock();
    private boolean closed;

    private OnnxModel(OrtEnvironment environment, OrtSession session, long padTokenId) {
        this.environment = environment;
        this.session = session;
        this.padTokenId = padTokenId;
    }

    /** Loads a model with default session options and pad token id 0. */
    public static OnnxModel load(Path modelFile) {
        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            return load(modelFile, options, 0L);
        }
    }

    /**
     * Loads a model. The caller keeps ownership of {@code options}, which may be
     * closed once this method returns.
     */
    public static OnnxModel load(Path modelFile, OrtSession.SessionOptions options, long padTokenId) {
        OrtEnvironment environment = OrtEnvironment.getEnvironment();
        OrtSession session;
        try {
            session = environment.createSession(modelFile.toString(), options);
        } catch (OrtException e) {
            throw new InferenceException("Failed to load " + modelFile, e);
        }
        try {
            requireNames("input", session.getInputNames(), INPUT_IDS, ATTENTION_MASK);
            requireNames("output", session.getOutputNames(), LOGITS);
        } catch (RuntimeException e) {
            closeQuietly(session, e);
            throw e;
        }
        return new OnnxModel(environment, session, padTokenId);
    }

    @Override
    public float[][] run(List<ModelInput> batch) {
        if (batch.isEmpty()) {
            return new float[0][];
        }
        int rows = batch.size();
        int columns = 0;
        for (ModelInput input : batch) {
            columns = Math.max(columns, input.length());
        }
        if ((long) rows * columns > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Batch too large: " + rows + " x " + columns);
        }
        long[] ids = new long[rows * columns];
        long[] mask = new long[rows * columns];
        Arrays.fill(ids, padTokenId);
        for (int row = 0; row < rows; row++) {
            batch.get(row).copyTo(ids, mask, row * columns);
        }
        long[] shape = {rows, columns};

        lifecycle.readLock().lock();
        try {
            if (closed) {
                throw new IllegalStateException("Model is closed");
            }
            try (OnnxTensor idsTensor = OnnxTensor.createTensor(environment, LongBuffer.wrap(ids), shape);
                 OnnxTensor maskTensor = OnnxTensor.createTensor(environment, LongBuffer.wrap(mask), shape);
                 OrtSession.Result result = session.run(
                         Map.of(INPUT_IDS, idsTensor, ATTENTION_MASK, maskTensor), Set.of(LOGITS))) {
                return readLogits(result, rows);
            } catch (OrtException e) {
                throw new InferenceException("Inference failed for batch of " + rows, e);
            }
        } finally {
            lifecycle.readLock().unlock();
        }
    }

    @Override
    public void close() {
        lifecycle.writeLock().lock();
        try {
            if (closed) {
                return;
            }
            closed = true;
            session.close();
        } catch (OrtException e) {
            throw new InferenceException("Failed to close session", e);
        } finally {
            lifecycle.writeLock().unlock();
        }
    }

    private static float[][] readLogits(OrtSession.Result result, int rows) {
        OnnxValue value = result.get(LOGITS)
                .orElseThrow(() -> new InferenceException("Missing output " + LOGITS));
        if (!(value instanceof OnnxTensor tensor)) {
            throw new InferenceException("Output " + LOGITS + " is not a tensor");
        }
        long[] shape = tensor.getInfo().getShape();
        if (shape.length != 2 || shape[0] != rows) {
            throw new InferenceException(
                    "Expected " + LOGITS + " shape [" + rows + ", classes], got " + Arrays.toString(shape));
        }
        FloatBuffer buffer = tensor.getFloatBuffer();
        if (buffer == null) {
            throw new InferenceException("Output " + LOGITS + " is not float32");
        }
        float[][] logits = new float[rows][(int) shape[1]];
        for (float[] row : logits) {
            buffer.get(row);
        }
        return logits;
    }

    private static void requireNames(String kind, Set<String> actual, String... expected) {
        for (String name : expected) {
            if (!actual.contains(name)) {
                throw new InferenceException("Model is missing " + kind + " " + name + "; found " + actual);
            }
        }
    }

    private static void closeQuietly(OrtSession session, Throwable failure) {
        try {
            session.close();
        } catch (OrtException e) {
            failure.addSuppressed(e);
        }
    }
}
