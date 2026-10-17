package lol.utf.themis.onnx;

import ai.onnxruntime.OrtSession;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.RejectedExecutionException;

/**
 * Moderates text with an exported model bundle. Equivalent to the Python
 * {@code Moderation.predict}: texts are tokenized like the slow DeBERTa
 * tokenizer, truncated to the bundle's {@code max_length}, and scored with
 * independent category and action softmaxes.
 *
 * <p>All methods are thread safe. Requests from every thread share one FIFO
 * queue served by a single inference thread, which starts on a request as soon
 * as it is free and never waits for a batch to fill. With
 * {@link Scheduling#batched(int)}, requests that queued while a call was running
 * share the next call. Tokenization runs on the calling thread.
 *
 * <p>Futures complete on the inference thread: use the {@code *Async} methods
 * of {@link CompletableFuture} for expensive follow-up work. Blocking methods
 * called from the inference thread throw {@link IllegalStateException} rather
 * than deadlock.
 *
 * <pre>{@code
 * try (Moderator moderator = Moderator.load(bundle, Scheduling.batched(16))) {
 *     Moderation result = moderator.moderate("hello world");
 *     if (result.action() == Action.BLOCK_AND_REVIEW) { ... }
 * }
 * }</pre>
 */
public final class Moderator implements AutoCloseable {

    private final DebertaTokenizer tokenizer;
    private final OnnxModel model;
    private final InferenceExecutor executor;
    private final int categoryStart;
    private final int actionStart;

    private Moderator(ModelBundle bundle, OnnxModel model, Scheduling scheduling) {
        this.tokenizer = bundle.tokenizer();
        this.model = model;
        this.categoryStart = bundle.categoryStart();
        this.actionStart = bundle.actionStart();
        this.executor = InferenceExecutor.create(model, scheduling);
    }

    /** Loads a bundle with FIFO scheduling. */
    public static Moderator load(Path bundleDirectory) {
        return load(bundleDirectory, Scheduling.fifo());
    }

    public static Moderator load(Path bundleDirectory, Scheduling scheduling) {
        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            return load(bundleDirectory, scheduling, options);
        }
    }

    /**
     * Loads a bundle with custom ONNX Runtime session options, for example
     * {@code setIntraOpNumThreads(1)} to keep each model call on one core. The
     * caller keeps ownership of {@code sessionOptions}.
     *
     * @throws InferenceException if the bundle is missing, invalid or incompatible
     */
    public static Moderator load(Path bundleDirectory, Scheduling scheduling,
                                 OrtSession.SessionOptions sessionOptions) {
        Objects.requireNonNull(scheduling, "scheduling");
        ModelBundle bundle;
        try {
            bundle = ModelBundle.read(bundleDirectory);
        } catch (IOException e) {
            throw new InferenceException("Failed to read model bundle " + bundleDirectory, e);
        }
        OnnxModel model = OnnxModel.load(bundle.modelFile(), sessionOptions, bundle.padTokenId());
        return new Moderator(bundle, model, scheduling);
    }

    /**
     * Moderates one text, blocking until its result is ready.
     *
     * @throws RejectedExecutionException if the moderator is closed or its queue is full
     * @throws InferenceException if the model fails
     * @throws IllegalStateException if called from the inference thread
     */
    public Moderation moderate(String text) {
        requireNotWorker();
        return await(submit(text));
    }

    /** Moderates texts, blocking until all results are ready; results are in input order. */
    public List<Moderation> moderateAll(List<String> texts) {
        requireNotWorker();
        return await(submitAll(texts));
    }

    /**
     * Queues one text. The future fails with {@link InferenceException} if the model fails.
     *
     * @throws RejectedExecutionException if the moderator is closed or its queue is full
     */
    public CompletableFuture<Moderation> submit(String text) {
        Objects.requireNonNull(text, "text");
        return executor.submit(tokenizer.encode(text)).thenApply(this::decode);
    }

    /**
     * Queues texts together so they can share batches; results are in input
     * order. Either every text is queued or none is.
     *
     * @throws RejectedExecutionException if the moderator is closed or its queue
     *     lacks room for every text
     */
    public CompletableFuture<List<Moderation>> submitAll(List<String> texts) {
        List<ModelInput> inputs = new ArrayList<>(texts.size());
        for (String text : texts) {
            inputs.add(tokenizer.encode(Objects.requireNonNull(text, "text")));
        }
        return executor.submitAll(inputs).thenApply(rows -> rows.stream().map(this::decode).toList());
    }

    /**
     * Finishes queued requests, then releases the model.
     *
     * @throws IllegalStateException if called from the inference thread; use an
     *     {@code *Async} stage to close from a future callback
     */
    @Override
    public void close() {
        requireNotWorker();
        try {
            executor.close();
        } finally {
            model.close();
        }
    }

    private Moderation decode(float[] logits) {
        int categories = Category.values().length;
        int actions = Action.values().length;
        if (logits.length < Math.max(categoryStart + categories, actionStart + actions)) {
            throw new InferenceException("Model returned " + logits.length + " logits per text");
        }
        return Moderation.of(softmax(logits, categoryStart, categories), softmax(logits, actionStart, actions));
    }

    static double[] softmax(float[] logits, int start, int count) {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = start; i < start + count; i++) {
            max = Math.max(max, logits[i]);
        }
        double[] probabilities = new double[count];
        double sum = 0;
        for (int i = 0; i < count; i++) {
            probabilities[i] = Math.exp(logits[start + i] - max);
            sum += probabilities[i];
        }
        for (int i = 0; i < count; i++) {
            probabilities[i] /= sum;
        }
        return probabilities;
    }

    private void requireNotWorker() {
        if (executor.isWorkerThread()) {
            throw new IllegalStateException(
                    "Blocking operations from the inference thread would deadlock; use submit and *Async stages");
        }
    }

    private static <T> T await(CompletableFuture<T> future) {
        try {
            return future.join();
        } catch (CompletionException e) {
            Throwable cause = e.getCause() == null ? e : e.getCause();
            throw new InferenceException("Moderation failed: " + cause.getMessage(), cause);
        }
    }
}
