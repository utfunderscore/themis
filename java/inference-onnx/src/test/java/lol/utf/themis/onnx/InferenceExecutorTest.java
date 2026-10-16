package lol.utf.themis.onnx;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

@Timeout(30)
class InferenceExecutorTest {

    /** Returns {first id, length, batch size} per row and records every batch. */
    private static final class FakeModel implements InferenceModel {
        final List<List<Long>> batches = new CopyOnWriteArrayList<>();
        final CountDownLatch entered = new CountDownLatch(1);
        final CountDownLatch release;
        final AtomicBoolean fail = new AtomicBoolean();
        final Set<Thread> threads = ConcurrentHashMap.newKeySet();
        final AtomicInteger active = new AtomicInteger();
        final AtomicInteger maxActive = new AtomicInteger();

        FakeModel(boolean blocked) {
            release = new CountDownLatch(blocked ? 1 : 0);
        }

        @Override
        public float[][] run(List<ModelInput> batch) {
            threads.add(Thread.currentThread());
            maxActive.accumulateAndGet(active.incrementAndGet(), Math::max);
            try {
                return compute(batch);
            } finally {
                active.decrementAndGet();
            }
        }

        private float[][] compute(List<ModelInput> batch) {
            batches.add(batch.stream().map(input -> input.inputIds()[0]).toList());
            entered.countDown();
            try {
                release.await();
            } catch (InterruptedException e) {
                throw new IllegalStateException(e);
            }
            if (fail.get()) {
                throw new InferenceException("boom");
            }
            float[][] logits = new float[batch.size()][];
            for (int i = 0; i < logits.length; i++) {
                ModelInput input = batch.get(i);
                logits[i] = new float[] {input.inputIds()[0], input.length(), batch.size()};
            }
            return logits;
        }

        List<Integer> sizes() {
            return batches.stream().map(List::size).toList();
        }

        List<Long> order() {
            return batches.stream().flatMap(List::stream).toList();
        }
    }

    private static ModelInput input(long id) {
        return ModelInput.of(id, 7, 8);
    }

    private static List<ModelInput> inputs(long from, long to) {
        List<ModelInput> inputs = new ArrayList<>();
        for (long id = from; id < to; id++) {
            inputs.add(input(id));
        }
        return inputs;
    }

    @Test
    void fifoRunsEachRequestAloneInSubmissionOrder() throws Exception {
        FakeModel model = new FakeModel(false);
        List<CompletableFuture<float[]>> futures = new ArrayList<>();
        try (InferenceExecutor executor = InferenceExecutor.fifo(model)) {
            for (ModelInput input : inputs(0, 50)) {
                futures.add(executor.submit(input));
            }
        }
        for (int i = 0; i < futures.size(); i++) {
            assertArrayEquals(new float[] {i, 3, 1}, futures.get(i).get());
        }
        assertTrue(model.sizes().stream().allMatch(size -> size == 1));
        assertEquals(inputs(0, 50).stream().map(input -> input.inputIds()[0]).toList(), model.order());
    }

    @Test
    void batchedGroupsQueuedRequestsInOrder() throws Exception {
        FakeModel model = new FakeModel(true);
        try (InferenceExecutor executor = InferenceExecutor.batched(model, 4)) {
            CompletableFuture<float[]> first = executor.submit(input(0));
            model.entered.await();
            List<CompletableFuture<float[]>> rest = new ArrayList<>();
            for (ModelInput input : inputs(1, 11)) {
                rest.add(executor.submit(input));
            }
            model.release.countDown();
            assertArrayEquals(new float[] {0, 3, 1}, first.get());
            for (int i = 0; i < rest.size(); i++) {
                assertEquals(i + 1, rest.get(i).get()[0]);
            }
        }
        assertEquals(List.of(1, 4, 4, 2), model.sizes());
        assertEquals(List.of(0L, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, 10L), model.order());
    }

    @Test
    void runsQueuedRequestsImmediatelyWithoutWaitingForBatch() throws Exception {
        FakeModel model = new FakeModel(false);
        try (InferenceExecutor executor = InferenceExecutor.batched(model, 64)) {
            assertArrayEquals(new float[] {0, 3, 1}, executor.submit(input(0)).get(5, TimeUnit.SECONDS));
            List<float[]> logits = executor.submitAll(inputs(1, 4)).get(5, TimeUnit.SECONDS);
            for (int i = 0; i < 3; i++) {
                assertArrayEquals(new float[] {i + 1, 3, 3}, logits.get(i));
            }
        }
        assertEquals(List.of(1, 3), model.sizes());
    }

    @Test
    void concurrentSubmittersReceiveTheirOwnResults() throws Exception {
        FakeModel model = new FakeModel(false);
        Scheduling config = Scheduling.batched(16).withQueueCapacity(10_000);
        ExecutorService clients = Executors.newFixedThreadPool(8);
        try (InferenceExecutor executor = InferenceExecutor.create(model, config)) {
            List<Future<?>> tasks = new ArrayList<>();
            for (int client = 0; client < 8; client++) {
                long base = client * 1000L;
                tasks.add(clients.submit(() -> {
                    List<CompletableFuture<float[]>> futures = new ArrayList<>();
                    for (long id = base; id < base + 200; id++) {
                        futures.add(executor.submit(input(id)));
                    }
                    for (int i = 0; i < futures.size(); i++) {
                        assertEquals(base + i, (long) futures.get(i).join()[0]);
                    }
                    return null;
                }));
            }
            for (Future<?> task : tasks) {
                task.get();
            }
        } finally {
            clients.shutdownNow();
        }
        assertEquals(1600, model.order().size());
        assertTrue(model.sizes().stream().allMatch(size -> size >= 1 && size <= 16));
        assertEquals(1, model.threads.size());
        assertEquals(1, model.maxActive.get());
    }

    @Test
    void completesFuturesOnTheWorkerThread() throws Exception {
        FakeModel model = new FakeModel(true);
        try (InferenceExecutor executor = InferenceExecutor.fifo(model)) {
            assertFalse(executor.isWorkerThread());
            CompletableFuture<Boolean> onWorker =
                    executor.submit(input(0)).thenApply(ignored -> executor.isWorkerThread());
            model.release.countDown();
            assertTrue(onWorker.get());
        }
    }

    @Test
    void modelFailureFailsBatchAndExecutorRecovers() throws Exception {
        FakeModel model = new FakeModel(false);
        try (InferenceExecutor executor = InferenceExecutor.fifo(model)) {
            model.fail.set(true);
            ExecutionException failure = assertThrows(ExecutionException.class,
                    () -> executor.submit(input(1)).get());
            assertInstanceOf(InferenceException.class, failure.getCause());
            model.fail.set(false);
            assertEquals(2, executor.submit(input(2)).get()[0]);
        }
    }

    @Test
    void rejectsWhenQueueIsFull() throws Exception {
        FakeModel model = new FakeModel(true);
        Scheduling config = Scheduling.fifo().withQueueCapacity(2);
        try (InferenceExecutor executor = InferenceExecutor.create(model, config)) {
            CompletableFuture<float[]> running = executor.submit(input(0));
            model.entered.await();
            assertThrows(RejectedExecutionException.class, () -> executor.submitAll(inputs(1, 4)));
            CompletableFuture<List<float[]>> queued = executor.submitAll(inputs(1, 3));
            assertThrows(RejectedExecutionException.class, () -> executor.submit(input(3)));
            model.release.countDown();
            running.get();
            assertEquals(2, queued.get().size());
        }
        assertEquals(List.of(0L, 1L, 2L), model.order());
    }

    @Test
    void closeDrainsQueueAndRejectsNewWork() throws Exception {
        FakeModel model = new FakeModel(true);
        InferenceExecutor executor = InferenceExecutor.batched(model, 8);
        CompletableFuture<List<float[]>> pending = executor.submitAll(inputs(0, 3));
        Thread closer = Thread.ofPlatform().start(executor::close);
        model.release.countDown();
        closer.join();
        assertEquals(3, pending.get(0, TimeUnit.SECONDS).size());
        assertThrows(RejectedExecutionException.class, () -> executor.submit(input(4)));
        executor.close();
    }

    @Test
    void cancelledRequestsAreSkipped() throws Exception {
        FakeModel model = new FakeModel(true);
        try (InferenceExecutor executor = InferenceExecutor.fifo(model)) {
            CompletableFuture<float[]> running = executor.submit(input(0));
            model.entered.await();
            CompletableFuture<float[]> cancelled = executor.submit(input(1));
            CompletableFuture<float[]> kept = executor.submit(input(2));
            cancelled.cancel(false);
            model.release.countDown();
            running.get();
            kept.get();
        }
        assertEquals(List.of(0L, 2L), model.order());
    }

    @Test
    void emptySubmitAllCompletesImmediately() throws Exception {
        try (InferenceExecutor executor = InferenceExecutor.fifo(new FakeModel(false))) {
            assertEquals(List.of(), executor.submitAll(List.of()).get());
        }
    }

    @Test
    void modelInputIsDefensivelyCopiedAndValidated() {
        long[] ids = {1, 2, 3};
        ModelInput input = ModelInput.of(ids);
        ids[0] = 99;
        input.inputIds()[1] = 99;
        assertArrayEquals(new long[] {1, 2, 3}, input.inputIds());
        assertArrayEquals(new long[] {1, 1, 1}, input.attentionMask());
        assertEquals(ModelInput.of(1, 2, 3), input);
        assertThrows(IllegalArgumentException.class, () -> ModelInput.of());
        assertThrows(IllegalArgumentException.class, () -> new ModelInput(new long[] {1}, new long[] {1, 1}));
        assertThrows(IllegalArgumentException.class, () -> Scheduling.batched(0));
    }
}
