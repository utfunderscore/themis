package lol.utf.themis.onnx;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Map;
import org.junit.jupiter.api.Test;

class ModerationTest {

    @Test
    void softmaxIsComputedPerGroup() {
        float[] logits = {0, 0, 1000, 1000, 0, (float) Math.log(3)};
        assertArrayEquals(new double[] {0.5, 0.5}, Moderator.softmax(logits, 2, 2), 1e-12);
        assertArrayEquals(new double[] {0.25, 0.75}, Moderator.softmax(logits, 4, 2), 1e-6);
    }

    @Test
    void picksMostProbableLabels() {
        Moderation result = Moderation.of(new double[] {0.1, 0.1, 0.5, 0.1, 0.1, 0.1}, new double[] {0.3, 0.7});
        assertEquals(Category.INSULT, result.category());
        assertEquals(Action.BLOCK_AND_REVIEW, result.action());
        assertEquals(0.5, result.probability(Category.INSULT));
        assertEquals(0.3, result.probability(Action.ALLOW));
        assertThrows(UnsupportedOperationException.class,
                () -> result.categoryProbabilities().put(Category.NONE, 1.0));
    }

    @Test
    void tiesPickFirstLabelLikePython() {
        Moderation result = Moderation.of(new double[] {0, 0.5, 0.5, 0, 0, 0}, new double[] {0.5, 0.5});
        assertEquals(Category.PROFANITY, result.category());
        assertEquals(Action.ALLOW, result.action());
    }

    @Test
    void requiresEveryLabel() {
        assertThrows(IllegalArgumentException.class, () -> new Moderation(
                Category.NONE, Action.ALLOW, Map.of(Category.NONE, 1.0), Map.of(Action.ALLOW, 1.0)));
    }
}
