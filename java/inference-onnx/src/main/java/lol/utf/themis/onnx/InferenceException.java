package lol.utf.themis.onnx;

/** Thrown when a model cannot be loaded or fails to run. */
public class InferenceException extends RuntimeException {

    public InferenceException(String message) {
        super(message);
    }

    public InferenceException(String message, Throwable cause) {
        super(message, cause);
    }
}
