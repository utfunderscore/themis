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
 * <p>Requests from any number of threads enter one bounded FIFO queue. Whenever
 * the worker is free it immediately takes what is queued, up to
 * {@link Scheduling#maxBatchSize()} requests, as one model call; it never waits
 * for a batch to fill. A request arriving at an idle executor therefore runs
 * alone and at once, while requests that queue up behind a running call share
 * the next one. Requests complete in submission order.
 *
 * <p>Futures are completed on the worker thread; use the {@code *Async} methods of
 * {@link CompletableFuture} for expensive follow-up work, and never block on a
 * request from the worker thread (see {@link #isWorkerThread()}). Requests whose
 * futures are cancelled before they are taken are skipped. The executor does not
 * close the model.
 */
final class InferenceExecutor implements AutoCloseable {

    private final InferenceModel model;
    private final Scheduling scheduling;
    private final ArrayDeque<Request> queue = new ArrayDeque<>();
    private final ReentrantLock lock = new ReentrantLock();
    private final Condition available = lock.newCondition();
    private final Thread worker;
    private boolean closed;

    private InferenceExecutor(InferenceModel model, Scheduling scheduling) {
        this.model = model;
        this.scheduling = scheduling;
        this.worker = Thread.ofPlatform().name("themis-onnx-inference").daemon(true).unstarted(this::work);
    }

    /** Runs requests one at a time in submission order. */
    public static InferenceExecutor fifo(InferenceModel model) {
        return create(model, Scheduling.fifo());
    }

    /** Runs up to {@code maxBatchSize} already-queued requests per model call. */
    public static InferenceExecutor batched(InferenceModel model, int maxBatchSize) {
        return create(model, Scheduling.batched(maxBatchSize));
    }

    public static InferenceExecutor create(InferenceModel model, Scheduling scheduling) {
        InferenceExecutor executor = new InferenceExecutor(
                Objects.requireNonNull(model, "model"), Objects.requireNonNull(scheduling, "scheduling"));
        executor.worker.start();
        return executor;
    }

    /** Whether the caller is the worker thread, which must not block on requests. */
    public boolean isWorkerThread() {
        return Thread.currentThread() == worker;
    }

    /**
     * Queues one input and returns a future for its logits.
     *
     * @throws RejectedExecutionException if the executor is closed or the queue is full
     */
    public CompletableFuture<float[]> submit(ModelInput input) {
        return enqueue(List.of(input)).getFirst();
    }

    /**
     * Queues inputs contiguously, so they can share batches, and returns their
     * logits in input order. Either every input is queued or none is.
     *
     * @throws RejectedExecutionException if the executor is closed or the queue
     *     lacks room for every input
     */
    public CompletableFuture<List<float[]>> submitAll(List<ModelInput> inputs) {
        List<CompletableFuture<float[]>> futures = enqueue(List.copyOf(inputs));
        return CompletableFuture.allOf(futures.toArray(CompletableFuture[]::new))
                .thenApply(ignored -> futures.stream().map(CompletableFuture::join).toList());
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

    private List<CompletableFuture<float[]>> enqueue(List<ModelInput> inputs) {
        List<CompletableFuture<float[]>> futures = new ArrayList<>(inputs.size());
        if (inputs.isEmpty()) {
            return futures;
        }
        lock.lock();
        try {
            if (closed) {
                throw new RejectedExecutionException("Executor is closed");
            }
            if (inputs.size() > scheduling.queueCapacity() - queue.size()) {
                throw new RejectedExecutionException(
                        "Queue is full: " + queue.size() + " of " + scheduling.queueCapacity()
                                + " used, " + inputs.size() + " requested");
            }
            for (ModelInput input : inputs) {
                CompletableFuture<float[]> future = new CompletableFuture<>();
                queue.addLast(new Request(input, future));
                futures.add(future);
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
            int size = Math.min(scheduling.maxBatchSize(), queue.size());
            List<Request> batch = new ArrayList<>(size);
            for (int i = 0; i < size; i++) {
                Request request = queue.pollFirst();
                if (!request.future().isDone()) {
                    batch.add(request);
                }
            }
            return batch;
        } finally {
            lock.unlock();
        }
    }

    private void execute(List<Request> batch) {
        List<ModelInput> inputs = new ArrayList<>(batch.size());
        for (Request request : batch) {
            inputs.add(request.input());
        }
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
