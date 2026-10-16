package lol.utf.themis.onnx;

import java.util.Arrays;
import java.util.Map;

/**
 * Matches the slow Hugging Face {@code DebertaV2Tokenizer} for single texts:
 * special tokens written in the text are kept whole, the remaining segments are
 * SentencePiece-encoded, and the result is right-truncated and wrapped as
 * {@code [CLS] ... [SEP]}. Immutable and thread safe.
 */
final class DebertaTokenizer {

    private final SentencePiece sentencePiece;
    private final Map<String, Integer> specialTokens;
    private final int clsId;
    private final int sepId;
    private final int maxLength;

    DebertaTokenizer(SentencePiece sentencePiece, Map<String, Integer> specialTokens,
                     int clsId, int sepId, int maxLength) {
        if (maxLength < 2) {
            throw new IllegalArgumentException("maxLength must fit [CLS] and [SEP]");
        }
        for (String token : specialTokens.keySet()) {
            if (token.isEmpty()) {
                throw new IllegalArgumentException("Special tokens must not be empty");
            }
        }
        this.sentencePiece = sentencePiece;
        this.specialTokens = Map.copyOf(specialTokens);
        this.clsId = clsId;
        this.sepId = sepId;
        this.maxLength = maxLength;
    }

    int maxLength() {
        return maxLength;
    }

    ModelInput encode(String text) {
        int limit = maxLength - 2;
        long[] ids = new long[maxLength];
        int count = 1;
        ids[0] = clsId;
        int position = 0;
        while (position < text.length() && count - 1 < limit) {
            int matchStart = text.length();
            String match = null;
            for (String token : specialTokens.keySet()) {
                int index = text.indexOf(token, position);
                if (index >= 0 && (index < matchStart
                        || index == matchStart && token.length() > match.length())) {
                    matchStart = index;
                    match = token;
                }
            }
            if (matchStart > position) {
                for (int id : sentencePiece.encode(text.substring(position, matchStart))) {
                    if (count - 1 == limit) {
                        break;
                    }
                    ids[count++] = id;
                }
            }
            if (match == null) {
                break;
            }
            if (count - 1 < limit) {
                ids[count++] = specialTokens.get(match);
            }
            position = matchStart + match.length();
        }
        ids[count++] = sepId;
        return ModelInput.of(Arrays.copyOf(ids, count));
    }
}
