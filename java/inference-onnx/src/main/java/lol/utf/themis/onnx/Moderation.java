package lol.utf.themis.onnx;

import java.util.Collections;
import java.util.EnumMap;
import java.util.Map;

/**
 * Moderation result for one text. Category and action probabilities are
 * independent distributions, each summing to 1; {@code category} and
 * {@code action} are their most probable labels.
 */
public record Moderation(
        Category category,
        Action action,
        Map<Category, Double> categoryProbabilities,
        Map<Action, Double> actionProbabilities) {

    public Moderation {
        categoryProbabilities = Collections.unmodifiableMap(complete(categoryProbabilities, Category.class));
        actionProbabilities = Collections.unmodifiableMap(complete(actionProbabilities, Action.class));
    }

    public double probability(Category category) {
        return categoryProbabilities.get(category);
    }

    public double probability(Action action) {
        return actionProbabilities.get(action);
    }

    /** Builds a result from softmax probabilities in label order. */
    static Moderation of(double[] categories, double[] actions) {
        Map<Category, Double> categoryMap = new EnumMap<>(Category.class);
        for (Category category : Category.values()) {
            categoryMap.put(category, categories[category.ordinal()]);
        }
        Map<Action, Double> actionMap = new EnumMap<>(Action.class);
        for (Action action : Action.values()) {
            actionMap.put(action, actions[action.ordinal()]);
        }
        return new Moderation(
                Category.values()[argmax(categories)], Action.values()[argmax(actions)],
                categoryMap, actionMap);
    }

    private static <K extends Enum<K>> EnumMap<K, Double> complete(Map<K, Double> probabilities, Class<K> type) {
        EnumMap<K, Double> copy = new EnumMap<>(type);
        copy.putAll(probabilities);
        if (copy.size() != type.getEnumConstants().length || copy.containsValue(null)) {
            throw new IllegalArgumentException("Expected a probability for every " + type.getSimpleName());
        }
        return copy;
    }

    private static int argmax(double[] values) {
        int best = 0;
        for (int i = 1; i < values.length; i++) {
            if (values[i] > values[best]) {
                best = i;
            }
        }
        return best;
    }
}
