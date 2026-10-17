package lol.utf.themis.onnx;

import java.util.Collections;
import java.util.EnumMap;
import java.util.Map;

/**
 * The category, action, and probabilities for a single text.
 *
 * <p>Results returned by {@link Moderator} contain the most probable category
 * and action. Use {@link #category()} and {@link #action()} to read these labels,
 * or {@code probability(...)} to look up the probability of a specific label.
 *
 * <pre>{@code
 * Moderation result = moderator.moderate("hello world");
 * Action action = result.action();
 * double threatProbability = result.probability(Category.THREAT);
 * }</pre>
 *
 * <p>Both probability maps are copied on construction and are unmodifiable.
 * Each map must contain a non-null value for every constant in its enum.
 * The constructor does not normalize probabilities, check their range or sum,
 * or check that the supplied category and action have the highest probabilities.
 *
 * @param category the selected content category
 * @param action the selected action
 * @param categoryProbabilities the probability for every category
 * @param actionProbabilities the probability for every action
 */
public record Moderation(
        Category category,
        Action action,
        Map<Category, Double> categoryProbabilities,
        Map<Action, Double> actionProbabilities) {

    /**
     * Creates a result with the supplied labels and copies of the probability maps.
     *
     * @throws IllegalArgumentException if either map is missing an enum constant
     *     or contains a null value
     * @throws NullPointerException if either map is null
     */
    public Moderation {
        categoryProbabilities = Collections.unmodifiableMap(complete(categoryProbabilities, Category.class));
        actionProbabilities = Collections.unmodifiableMap(complete(actionProbabilities, Action.class));
    }

    /**
     * Returns the probability stored for the given category.
     *
     * @param category the category to look up
     * @return the stored probability
     * @throws NullPointerException if {@code category} is null
     */
    public double probability(Category category) {
        return categoryProbabilities.get(category);
    }

    /**
     * Returns the probability stored for the given action.
     *
     * @param action the action to look up
     * @return the stored probability
     * @throws NullPointerException if {@code action} is null
     */
    public double probability(Action action) {
        return actionProbabilities.get(action);
    }

    /**
     * Builds a result from probability arrays indexed by enum ordinal, selecting
     * the label with the highest value in each array. Ties use the first label
     * in enum declaration order. Each array must have one value per enum constant.
     */
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
