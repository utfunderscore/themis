package lol.utf.themis.onnx;

/** Recommended handling, predicted independently of the {@link Category}. */
public enum Action {
    ALLOW("allow"),
    /** Block the message and send it for human review. */
    BLOCK_AND_REVIEW("block_and_review");

    private final String label;

    Action(String label) {
        this.label = label;
    }

    /** The label used by the training data and model bundle. */
    public String label() {
        return label;
    }
}
