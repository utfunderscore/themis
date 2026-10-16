package lol.utf.themis.onnx;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParseException;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.math.BigDecimal;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;

/**
 * A directory written by {@code onnx_export.py}: model.onnx, spm.model,
 * tokenizer_config.json and inference_config.json. Reading validates that the
 * bundle matches the labels, tokenizer and postprocessing implemented here.
 */
record ModelBundle(Path modelFile, DebertaTokenizer tokenizer, long padTokenId, int categoryStart, int actionStart) {

    static ModelBundle read(Path directory) throws IOException {
        try {
            JsonObject config = readJson(directory.resolve("inference_config.json"));
            requireValue(config, "format_version", 1);
            requireValue(config, "output", OnnxModel.LOGITS);
            requireValue(config, "postprocessing", "independent_softmax_per_group");
            if (!object(config, "inputs").keySet().equals(Set.of(OnnxModel.INPUT_IDS, OnnxModel.ATTENTION_MASK))) {
                throw invalid("inputs must be " + OnnxModel.INPUT_IDS + " and " + OnnxModel.ATTENTION_MASK);
            }
            JsonObject groups = object(config, "output_groups");
            int categoryStart = groupStart(object(groups, "category"), Stream.of(Category.values()).map(Category::label).toList());
            int actionStart = groupStart(object(groups, "action"), Stream.of(Action.values()).map(Action::label).toList());

            JsonObject tokenizer = object(config, "tokenizer");
            requireValue(tokenizer, "use_fast", false);
            requireValue(tokenizer, "padding_side", "right");
            requireValue(tokenizer, "truncation_side", "right");

            JsonObject tokenizerConfig = readJson(directory.resolve("tokenizer_config.json"));
            requireValue(tokenizerConfig, "tokenizer_class", "DebertaV2Tokenizer");
            requireOptional(tokenizerConfig, "do_lower_case", false);
            requireOptional(tokenizerConfig, "split_by_punct", false);

            DebertaTokenizer encoder = new DebertaTokenizer(
                    SentencePiece.load(directory.resolve("spm.model")),
                    specialTokens(object(tokenizerConfig, "added_tokens_decoder")),
                    integer(tokenizer, "cls_token_id"),
                    integer(tokenizer, "sep_token_id"),
                    integer(config, "max_length"));
            return new ModelBundle(directory.resolve("model.onnx"), encoder,
                    integer(tokenizer, "pad_token_id"), categoryStart, actionStart);
        } catch (IllegalArgumentException | IllegalStateException | JsonParseException e) {
            throw new InferenceException("Invalid model bundle " + directory + ": " + e.getMessage(), e);
        }
    }

    private static Map<String, Integer> specialTokens(JsonObject decoder) {
        Map<String, Integer> tokens = new HashMap<>();
        for (Map.Entry<String, JsonElement> entry : decoder.entrySet()) {
            JsonObject token = entry.getValue().getAsJsonObject();
            for (String flag : List.of("lstrip", "rstrip", "single_word")) {
                requireOptional(token, flag, false);
            }
            tokens.put(string(token, "content"), Integer.parseInt(entry.getKey()));
        }
        return tokens;
    }

    private static int groupStart(JsonObject group, List<String> expected) {
        int start = integer(group, "start");
        List<String> labels = new ArrayList<>();
        for (JsonElement label : group.getAsJsonArray("labels")) {
            labels.add(label.getAsString());
        }
        if (!labels.equals(expected) || integer(group, "end") - start != expected.size() || start < 0) {
            throw invalid("output group " + labels + " must have labels " + expected);
        }
        return start;
    }

    private static JsonObject readJson(Path file) throws IOException {
        return JsonParser.parseString(Files.readString(file)).getAsJsonObject();
    }

    private static JsonObject object(JsonObject parent, String name) {
        JsonElement value = parent.get(name);
        if (value == null || !value.isJsonObject()) {
            throw invalid(name + " must be an object");
        }
        return value.getAsJsonObject();
    }

    private static int integer(JsonObject parent, String name) {
        JsonElement value = parent.get(name);
        if (value == null || !value.isJsonPrimitive() || !value.getAsJsonPrimitive().isNumber()) {
            throw invalid(name + " must be an integer");
        }
        return value.getAsInt();
    }

    private static String string(JsonObject parent, String name) {
        JsonElement value = parent.get(name);
        if (value == null || !value.isJsonPrimitive() || !value.getAsJsonPrimitive().isString()) {
            throw invalid(name + " must be a string");
        }
        return value.getAsString();
    }

    private static void requireValue(JsonObject parent, String name, Object expected) {
        if (!parent.has(name) || !matches(parent.get(name), expected)) {
            throw invalid(name + " must be " + expected + ", got " + parent.get(name));
        }
    }

    private static void requireOptional(JsonObject parent, String name, Object expected) {
        if (parent.has(name) && !parent.get(name).isJsonNull() && !matches(parent.get(name), expected)) {
            throw invalid(name + " must be " + expected + ", got " + parent.get(name));
        }
    }

    private static boolean matches(JsonElement value, Object expected) {
        if (!value.isJsonPrimitive()) {
            return false;
        }
        return switch (expected) {
            case Boolean b -> value.getAsJsonPrimitive().isBoolean() && value.getAsBoolean() == b;
            case Integer i -> value.getAsJsonPrimitive().isNumber() && value.getAsBigDecimal().compareTo(BigDecimal.valueOf(i)) == 0;
            case String s -> value.getAsJsonPrimitive().isString() && value.getAsString().equals(s);
            default -> throw new IllegalArgumentException("Unsupported expected value " + expected);
        };
    }

    private static IllegalArgumentException invalid(String message) {
        return new IllegalArgumentException(message);
    }
}
