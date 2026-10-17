package lol.utf.themis.onnx;

import ai.onnxruntime.OrtSession;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.function.Supplier;

/**
 * Moderates text with an exported model bundle. Equivalent to the Python
 * {@code Moderation.predict}: texts are tokenized like the slow DeBERTa
 * tokenizer, truncated to the bundle's {@code max_length}, and scored with
 * independent category and action softmaxes.
 *
 * <p>All methods are thread safe. Requests from every thread share one queue,
 * served in submission order by a single inference thread. Requests that
 * queued while a call was running share the next call, up to
 * {@code maxBatchSize}. Tokenization runs on the calling thread, so it runs in
 * parallel across callers and never delays other requests; it stops once
 * {@code max_length} tokens are known, so long texts stay cheap.
 *
 * <p>No method throws. Every failure, including bad arguments, a closed
 * moderator, an invalid bundle and a failing model, arrives through the returned
 * {@link CompletableFuture}. Stages derived from a future (for example with
 * {@code thenApply}) see failures wrapped in a {@link CompletionException}, as
 * usual for {@link CompletableFuture}.
 *
 * <p>Result futures complete on the inference thread: use the {@code *Async}
 * methods of {@link CompletableFuture} for expensive follow-up work, and never
 * block on a result from a callback.
 *
 * <pre>{@code
 * Moderator.load(bundle, 16).thenCompose(moderator ->
 *         moderator.moderate("hello world")
 *                 .whenComplete((result, failure) -> moderator.closeAsync()));
 * }</pre>
 */
public final class Moderator implements AutoCloseable {

    private static final System.Logger LOGGER = System.getLogger(Moderator.class.getName());

    private final DebertaTokenizer tokenizer;
    private final OnnxModel model;
    private final InferenceExecutor executor;
    private final int categoryStart;
    private final int actionStart;

    private Moderator(ModelBundle bundle, OnnxModel model, int maxBatchSize) {
        this.tokenizer = bundle.tokenizer();
        this.model = model;
        this.categoryStart = bundle.categoryStart();
        this.actionStart = bundle.actionStart();
        this.executor = InferenceExecutor.create(model, maxBatchSize);
    }

    /** Loads a bundle that runs one request per model call. */
    public static CompletableFuture<Moderator> load(Path bundleDirectory) {
        return load(bundleDirectory, 1);
    }

    /**
     * Loads a bundle that runs up to {@code maxBatchSize} queued requests per
     * model call. It never waits for a batch to fill. Loading runs on a
     * background thread.
     *
     * <p>The future fails with {@link InferenceException} if the bundle is missing,
     * invalid or incompatible, and with {@link IllegalArgumentException} if
     * {@code maxBatchSize} is not positive.
     */
    public static CompletableFuture<Moderator> load(Path bundleDirectory, int maxBatchSize) {
        if (maxBatchSize < 1) {
            return CompletableFuture.failedFuture(new IllegalArgumentException("maxBatchSize must be positive"));
        }
        return background(() -> {
            try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
                return create(bundleDirectory, maxBatchSize, options);
            }
        });
    }

    /**
     * Loads a bundle with custom ONNX Runtime session options, for example
     * {@code setIntraOpNumThreads(1)} to keep each model call on one core. The
     * caller keeps ownership of {@code sessionOptions} and must keep it open
     * until the returned future completes.
     *
     * <p>The future fails with {@link InferenceException} if the bundle is missing,
     * invalid or incompatible, and with {@link IllegalArgumentException} if
     * {@code maxBatchSize} is not positive.
     */
    public static CompletableFuture<Moderator> load(Path bundleDirectory, int maxBatchSize,
                                                    OrtSession.SessionOptions sessionOptions) {
        if (maxBatchSize < 1) {
            return CompletableFuture.failedFuture(new IllegalArgumentException("maxBatchSize must be positive"));
        }
        return background(() -> create(bundleDirectory, maxBatchSize, sessionOptions));
    }

    private static Moderator create(Path bundleDirectory, int maxBatchSize, OrtSession.SessionOptions options) {
        ModelBundle bundle;
        try {
            bundle = ModelBundle.read(bundleDirectory);
        } catch (IOException e) {
            throw new InferenceException("Failed to read model bundle " + bundleDirectory, e);
        }
        OnnxModel model = OnnxModel.load(bundle.modelFile(), options, bundle.padTokenId());
        try {
            return new Moderator(bundle, model, maxBatchSize);
        } catch (RuntimeException | Error e) {
            try {
                model.close();
            } catch (RuntimeException suppressed) {
                e.addSuppressed(suppressed);
            }
            throw e;
        }
    }

    /**
     * Queues one text. The future fails with {@link InferenceException} if the
     * model fails, {@link java.util.concurrent.RejectedExecutionException} if the
     * moderator is closed, and {@link NullPointerException} if {@code text} is null.
     */
    public CompletableFuture<Moderation> moderate(String text) {
        try {
            Objects.requireNonNull(text, "text");
            return executor.submit(tokenizer.encode(text)).thenApply(this::decode);
        } catch (RuntimeException e) {
            return CompletableFuture.failedFuture(e);
        }
    }

    /**
     * Queues texts together so they can share batches; results are in input
     * order. Either every text is queued or none is. Failures are reported as for
     * {@link #moderate}, and a null list or element fails the future with
     * {@link NullPointerException}.
     */
    public CompletableFuture<List<Moderation>> moderateAll(List<String> texts) {
        try {
            List<ModelInput> inputs = new ArrayList<>(texts.size());
            for (String text : texts) {
                inputs.add(tokenizer.encode(Objects.requireNonNull(text, "text")));
            }
            return executor.submitAll(inputs).thenApply(rows -> rows.stream().map(this::decode).toList());
        } catch (RuntimeException e) {
            return CompletableFuture.failedFuture(e);
        }
    }

    /**
     * Stops accepting requests, finishes the queued ones, then releases the model,
     * on a background thread. Safe to call from a future callback, but do not block
     * on the returned future there. The future fails with {@link InferenceException}
     * if the model cannot be released.
     */
    public CompletableFuture<Void> closeAsync() {
        return background(() -> {
            try {
                executor.close();
            } finally {
                model.close();
            }
            return null;
        });
    }

    /**
     * Closes like {@link #closeAsync()} and waits for it, so it can be used with
     * try-with-resources. It never throws: a failure to release the model is logged.
     * When called from a future callback it does not wait, since that would deadlock.
     */
    @Override
    public void close() {
        CompletableFuture<Void> closing = closeAsync().exceptionally(failure -> {
            LOGGER.log(System.Logger.Level.WARNING, "Failed to close moderator", failure);
            return null;
        });
        if (!executor.isWorkerThread()) {
            closing.join();
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

    /** Runs the task on its own daemon thread and reports its result or failure through the future. */
    private static <T> CompletableFuture<T> background(Supplier<T> task) {
        CompletableFuture<T> future = new CompletableFuture<>();
        try {
            Thread.ofPlatform().name("themis-onnx-background").daemon().start(() -> {
                try {
                    future.complete(task.get());
                } catch (Throwable failure) {
                    future.completeExceptionally(failure);
                }
            });
        } catch (Throwable failure) {
            future.completeExceptionally(failure);
        }
        return future;
    }
}
