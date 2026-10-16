package lol.utf.themis.onnx;

import java.util.Arrays;

/**
 * One tokenized, unpadded sequence. Arrays are copied on construction and access,
 * so instances are immutable and safe to share between threads.
 */
record ModelInput(long[] inputIds, long[] attentionMask) {

    public ModelInput {
        if (inputIds.length == 0) {
            throw new IllegalArgumentException("inputIds must not be empty");
        }
        if (inputIds.length != attentionMask.length) {
            throw new IllegalArgumentException(
                    "inputIds and attentionMask lengths differ: "
                            + inputIds.length + " != " + attentionMask.length);
        }
        inputIds = inputIds.clone();
        attentionMask = attentionMask.clone();
    }

    /** Creates an input that attends to every token. */
    public static ModelInput of(long... inputIds) {
        long[] mask = new long[inputIds.length];
        Arrays.fill(mask, 1L);
        return new ModelInput(inputIds, mask);
    }

    public int length() {
        return inputIds.length;
    }

    @Override
    public long[] inputIds() {
        return inputIds.clone();
    }

    @Override
    public long[] attentionMask() {
        return attentionMask.clone();
    }

    void copyTo(long[] ids, long[] mask, int offset) {
        System.arraycopy(inputIds, 0, ids, offset, inputIds.length);
        System.arraycopy(attentionMask, 0, mask, offset, attentionMask.length);
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof ModelInput that
                && Arrays.equals(inputIds, that.inputIds)
                && Arrays.equals(attentionMask, that.attentionMask);
    }

    @Override
    public int hashCode() {
        return 31 * Arrays.hashCode(inputIds) + Arrays.hashCode(attentionMask);
    }

    @Override
    public String toString() {
        return "ModelInput[length=" + inputIds.length + "]";
    }
}
