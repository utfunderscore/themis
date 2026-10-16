package lol.utf.themis.onnx;

import java.util.List;

/**
 * Synchronous batch inference. Implementations must be safe to call from
 * multiple threads concurrently.
 */
interface InferenceModel {

    /**
     * Runs one batch and returns one row of logits per input, in input order.
     *
     * @throws InferenceException if the model fails to run
     */
    float[][] run(List<ModelInput> batch);
}
