package lol.utf.themis.onnx;

/** Content category predicted by the moderation model, in training label order. */
public enum Category {
    NONE("none"),
    PROFANITY("profanity"),
    INSULT("insult"),
    SLUR("slur"),
    SEXUAL("sexual"),
    THREAT("threat");

    private final String label;

    Category(String label) {
        this.label = label;
    }

    /** The label used by the training data and model bundle. */
    public String label() {
        return label;
    }
}
