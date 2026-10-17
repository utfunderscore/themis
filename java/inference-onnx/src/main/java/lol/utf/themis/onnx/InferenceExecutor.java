package lol.utf.themis.onnx;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Thread-safe asynchronous front end for an {@link InferenceModel}, served by a
 * single worker thread so the model is never called concurrently.
 *
 * <p>Requests from any number of threads enter one unbounded FIFO queue. Whenever
 * the worker is free it immediately takes what is queued, up to
 * {@code maxBatchSize} requests, as one model call; it never waits for a batch
 * to fill. A request arriving at an idle executor therefore runs alone and at
 * once, while requests that queue up behind a running call share the next one,
 * and a cap of 1 runs requests one at a time. Requests complete in submission
 * order.
 *
 * <p>When more than {@value #DEFAULT_QUEUE_WARNING_THRESHOLD} requests are
 * waiting (or the value of the {@value #QUEUE_WARNING_THRESHOLD_ENV}
 * environment variable), a warning is logged; it is logged again only after the
 * backlog has drained to half the threshold.
 *
 * <p>Futures are completed on the worker thread; use the {@code *Async} methods of
 * {@link CompletableFuture} for expensive follow-up work, and never block on a
 * request from the worker thread (see {@link #isWorkerThread()}). Requests whose
 * futures are cancelled before they are taken are skipped. The executor does not
 * close the model.
 */
final class InferenceExecutor implements AutoCloseable {

    static final String QUEUE_WARNING_THRESHOLD_ENV = "THEMIS_QUEUE_WARNING_THRESHOLD";
    static final int DEFAULT_QUEUE_WARNING_THRESHOLD = 1024;

    private static final System.Logger LOGGER = System.getLogger(InferenceExecutor.class.getName());
    private static final int QUEUE_WARNING_THRESHOLD =
            parseQueueWarningThreshold(System.getenv(QUEUE_WARNING_THRESHOLD_ENV));

    private final InferenceModel model;
    private final int maxBatchSize;
    private final int queueWarningThreshold;
    private final ArrayDeque<Request> queue = new ArrayDeque<>();
    private final ReentrantLock lock = new ReentrantLock();
    private final Condition available = lock.newCondition();
    private final Thread worker;
    private boolean closed;
    private boolean backlogWarned;

    private InferenceExecutor(InferenceModel model, int maxBatchSize, int queueWarningThreshold) {
        this.model = model;
        this.maxBatchSize = maxBatchSize;
        this.queueWarningThreshold = queueWarningThreshold;
        this.worker = Thread.ofPlatform().name("themis-onnx-inference").daemon(true).unstarted(this::work);
    }

    /**
     * Starts an executor that runs up to {@code maxBatchSize} queued requests per
     * model call.
     *
     * @throws IllegalArgumentException if {@code maxBatchSize} is not positive
     */
    public static InferenceExecutor create(InferenceModel model, int maxBatchSize) {
        return create(model, maxBatchSize, QUEUE_WARNING_THRESHOLD);
    }

    static InferenceExecutor create(InferenceModel model, int maxBatchSize, int queueWarningThreshold) {
        Objects.requireNonNull(model, "model");
        if (maxBatchSize < 1) {
            throw new IllegalArgumentException("maxBatchSize must be positive");
        }
        if (queueWarningThreshold < 1) {
            throw new IllegalArgumentException("queueWarningThreshold must be positive");
        }
        InferenceExecutor executor = new InferenceExecutor(model, maxBatchSize, queueWarningThreshold);
        executor.worker.start();
        return executor;
    }

    /** Parses a positive threshold, falling back to the default if unset or invalid. */
    static int parseQueueWarningThreshold(String value) {
        if (value == null || value.isBlank()) {
            return DEFAULT_QUEUE_WARNING_THRESHOLD;
        }
        try {
            int threshold = Integer.parseInt(value.strip());
            if (threshold >= 1) {
                return threshold;
            }
        } catch (NumberFormatException ignored) {
            // Falls through to the warning below.
        }
        LOGGER.log(System.Logger.Level.WARNING,
                "Ignoring {0}={1}: expected a positive integer, using {2}",
                QUEUE_WARNING_THRESHOLD_ENV, value, DEFAULT_QUEUE_WARNING_THRESHOLD);
        return DEFAULT_QUEUE_WARNING_THRESHOLD;
    }

    /** Whether the caller is the worker thread, which must not block on requests. */
    public boolean isWorkerThread() {
        return Thread.currentThread() == worker;
    }

    /**
     * Queues one request and returns a future for its logits. Never throws: the
     * future fails with {@link RejectedExecutionException} if the executor is
     * closed, or {@link NullPointerException} if the request is null.
     */
    public CompletableFuture<float[]> submit(ModelInput request) {
        try {
            return enqueue(List.of(request)).getFirst();
        } catch (RuntimeException e) {
            return CompletableFuture.failedFuture(e);
        }
    }

    /**
     * Queues requests contiguously, so they can share batches, and returns their
     * logits in request order. Either every request is queued or none is. Never
     * throws: failures are reported as for {@link #submit}.
     */
    public CompletableFuture<List<float[]>> submitAll(List<ModelInput> requests) {
        try {
            List<CompletableFuture<float[]>> futures = enqueue(List.copyOf(requests));
            return CompletableFuture.allOf(futures.toArray(CompletableFuture[]::new))
                    .thenApply(ignored -> futures.stream().map(CompletableFuture::join).toList());
        } catch (RuntimeException e) {
            return CompletableFuture.failedFuture(e);
        }
    }

    /**
     * Stops accepting requests, finishes every queued request, and waits for the
     * worker to exit. Calling this from the worker thread does not wait.
     */
    @Override
    public void close() {
        lock.lock();
        try {
            closed = true;
            available.signal();
        } finally {
            lock.unlock();
        }
        if (isWorkerThread()) {
            return;
        }
        boolean interrupted = false;
        while (true) {
            try {
                worker.join();
                break;
            } catch (InterruptedException e) {
                interrupted = true;
            }
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
    }

    private List<CompletableFuture<float[]>> enqueue(List<ModelInput> requests) {
        List<CompletableFuture<float[]>> futures = new ArrayList<>(requests.size());
        if (requests.isEmpty()) {
            return futures;
        }
        lock.lock();
        try {
            if (closed) {
                throw new RejectedExecutionException("Executor is closed");
            }
            for (ModelInput request : requests) {
                CompletableFuture<float[]> future = new CompletableFuture<>();
                queue.addLast(new Request(request, future));
                futures.add(future);
            }
            if (!backlogWarned && queue.size() > queueWarningThreshold) {
                backlogWarned = true;
                LOGGER.log(System.Logger.Level.WARNING,
                        "Inference queue has {0} waiting requests, above the warning threshold of {1}",
                        queue.size(), queueWarningThreshold);
            }
            available.signal();
        } finally {
            lock.unlock();
        }
        return futures;
    }

    private void work() {
        List<Request> batch;
        while ((batch = nextBatch()) != null) {
            if (!batch.isEmpty()) {
                execute(batch);
            }
        }
    }

    /** Returns the next batch, possibly empty if all were cancelled, or null on shutdown. */
    private List<Request> nextBatch() {
        lock.lock();
        try {
            while (queue.isEmpty()) {
                if (closed) {
                    return null;
                }
                available.awaitUninterruptibly();
            }
            int size = Math.min(maxBatchSize, queue.size());
            List<Request> batch = new ArrayList<>(size);
            for (int i = 0; i < size; i++) {
                Request request = queue.pollFirst();
                if (!request.future().isDone()) {
                    batch.add(request);
                }
            }
            if (queue.size() <= queueWarningThreshold / 2) {
                backlogWarned = false;
            }
            return batch;
        } finally {
            lock.unlock();
        }
    }

    private void execute(List<Request> batch) {
        List<ModelInput> inputs = batch.stream().map(Request::input).toList();
        float[][] logits;
        try {
            logits = model.run(inputs);
            if (logits.length != batch.size()) {
                throw new InferenceException(
                        "Model returned " + logits.length + " rows for batch of " + batch.size());
            }
        } catch (Throwable failure) {
            for (Request request : batch) {
                request.future().completeExceptionally(failure);
            }
            return;
        }
        for (int i = 0; i < batch.size(); i++) {
            batch.get(i).future().complete(logits[i]);
        }
    }

    private record Request(ModelInput input, CompletableFuture<float[]> future) {
    }
}
