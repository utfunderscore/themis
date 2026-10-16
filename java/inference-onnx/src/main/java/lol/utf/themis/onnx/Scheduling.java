package lol.utf.themis.onnx;

/**
 * How queued requests are grouped into model calls. A single worker serves
 * the queue in submission order and starts as soon as anything is queued.
 *
 * @param maxBatchSize most already-queued requests per model call; 1 disables batching
 * @param queueCapacity most requests waiting for the worker
 */
public record Scheduling(int maxBatchSize, int queueCapacity) {

    public static final int DEFAULT_QUEUE_CAPACITY = 1024;

    public Scheduling {
        if (maxBatchSize < 1) {
            throw new IllegalArgumentException("maxBatchSize must be positive");
        }
        if (queueCapacity < 1) {
            throw new IllegalArgumentException("queueCapacity must be positive");
        }
    }

    /** One request per model call, in submission order. */
    public static Scheduling fifo() {
        return new Scheduling(1, DEFAULT_QUEUE_CAPACITY);
    }

    /**
     * Up to {@code maxBatchSize} requests per model call, taken from whatever
     * queued while the previous call ran. Never waits for a batch to fill.
     */
    public static Scheduling batched(int maxBatchSize) {
        return new Scheduling(maxBatchSize, DEFAULT_QUEUE_CAPACITY);
    }

    public Scheduling withQueueCapacity(int queueCapacity) {
        return new Scheduling(maxBatchSize, queueCapacity);
    }
}
