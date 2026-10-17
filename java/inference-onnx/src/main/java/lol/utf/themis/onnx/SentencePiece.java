package lol.utf.themis.onnx;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import sentencepiece.SentencepieceModel.ModelProto;
import sentencepiece.SentencepieceModel.NormalizerSpec;
import sentencepiece.SentencepieceModel.TrainerSpec;

/**
 * SentencePiece unigram encoder matching the reference implementation's
 * Viterbi segmentation and byte fallback. Immutable and thread safe.
 */
final class SentencePiece {

    static final int NORMAL = 1;
    static final int UNKNOWN = 2;
    static final int CONTROL = 3;
    static final int USER_DEFINED = 4;
    static final int UNUSED = 5;
    static final int BYTE = 6;

    private static final float UNKNOWN_PENALTY = 10.0f;
    private static final char SPACE_SYMBOL = '\u2581';

    record Piece(String piece, float score, int type) {
    }

    private final SentencePieceNormalizer normalizer;
    private final Map<String, Integer> normalPieces = new HashMap<>();
    private final float[] scores;
    private final int[] byteIds = new int[256];
    private final int unknownId;
    private final float unknownScore;
    private final int maxPieceLength;
    /** Whether every space symbol starts a piece, so segmentation can stop there. */
    private final boolean spaceStartsPiece;

    SentencePiece(List<Piece> pieces, SentencePieceNormalizer normalizer) {
        this.normalizer = normalizer;
        this.scores = new float[pieces.size()];
        Arrays.fill(byteIds, -1);
        int unknown = -1;
        float minScore = Float.MAX_VALUE;
        int longest = 0;
        boolean spaceOnlyLeading = true;
        for (int id = 0; id < pieces.size(); id++) {
            Piece piece = pieces.get(id);
            scores[id] = piece.score();
            switch (piece.type()) {
                case NORMAL -> {
                    if (normalPieces.putIfAbsent(piece.piece(), id) != null) {
                        throw new IllegalArgumentException("Duplicate piece " + piece.piece());
                    }
                    minScore = Math.min(minScore, piece.score());
                    longest = Math.max(longest, piece.piece().length());
                    spaceOnlyLeading &= piece.piece().indexOf(SPACE_SYMBOL, 1) < 0;
                }
                case UNKNOWN -> unknown = id;
                case BYTE -> byteIds[parseByte(piece.piece())] = id;
                case CONTROL, UNUSED -> {
                }
                case USER_DEFINED -> throw new IllegalArgumentException("User-defined pieces are not supported");
                default -> throw new IllegalArgumentException("Unknown piece type " + piece.type());
            }
        }
        if (unknown < 0) {
            throw new IllegalArgumentException("Model has no unknown piece");
        }
        for (int b = 0; b < byteIds.length; b++) {
            if (byteIds[b] < 0) {
                throw new IllegalArgumentException("Model lacks byte fallback piece for " + b);
            }
        }
        this.unknownId = unknown;
        this.unknownScore = minScore - UNKNOWN_PENALTY;
        this.maxPieceLength = longest;
        this.spaceStartsPiece = spaceOnlyLeading;
    }

    /** Loads a unigram {@code .model} file trained with byte fallback. */
    static SentencePiece load(Path modelFile) throws IOException {
        ModelProto model = ModelProto.parseFrom(Files.readAllBytes(modelFile));
        TrainerSpec trainer = model.getTrainerSpec();
        // Proto2 preserves unknown enum values as unknown fields, while getters
        // return defaults. Reject them rather than accidentally accepting UNIGRAM.
        if (trainer.getUnknownFields().hasField(TrainerSpec.MODEL_TYPE_FIELD_NUMBER)
                || trainer.getModelType() != TrainerSpec.ModelType.UNIGRAM) {
            throw new IllegalArgumentException("Only unigram SentencePiece models are supported");
        }
        if (!trainer.getByteFallback()) {
            throw new IllegalArgumentException("Only SentencePiece models with byte fallback are supported");
        }
        if (trainer.getTreatWhitespaceAsSuffix()) {
            throw new IllegalArgumentException("treat_whitespace_as_suffix is not supported");
        }
        List<Piece> pieces = new ArrayList<>(model.getPiecesCount());
        for (ModelProto.SentencePiece piece : model.getPiecesList()) {
            if (piece.getUnknownFields().hasField(ModelProto.SentencePiece.TYPE_FIELD_NUMBER)) {
                throw new IllegalArgumentException("Unknown piece type");
            }
            pieces.add(new Piece(piece.getPiece(), piece.getScore(), piece.getType().getNumber()));
        }
        SentencePieceNormalizer normalizer = readNormalizer(model.getNormalizerSpec());
        return new SentencePiece(pieces, normalizer);
    }

    /** Encodes text to piece ids, without special tokens. */
    int[] encode(String text) {
        return encode(text, Integer.MAX_VALUE);
    }

    /**
     * Encodes text to piece ids, without special tokens, stopping once at least
     * {@code minTokens} ids are known. The result is always a prefix of
     * {@link #encode(String)} with at least {@code min(minTokens, total)} ids.
     *
     * <p>When no piece contains the space symbol after its first character, every
     * space symbol in the normalized text is a boundary of every segmentation.
     * The Viterbi scores up to a boundary do not depend on later text, so the
     * best path to the first boundary with enough ids is exactly the prefix of
     * the full best path, and the rest of the text can be skipped.
     */
    int[] encode(String text, int minTokens) {
        String normalized = normalizer.normalize(text);
        int length = normalized.length();
        if (length == 0) {
            return new int[0];
        }
        float[] bestScore = new float[length + 1];
        int[] bestStart = new int[length + 1];
        int[] bestId = new int[length + 1];
        int[] tokenCount = new int[length + 1];
        Arrays.fill(bestStart, -1);

        int stop = length;
        for (int start = 0; start < length; ) {
            int charLength = Character.charCount(normalized.codePointAt(start));
            if (start > 0) {
                // Every piece ending here has been scored, so the path to start is final.
                tokenCount[start] = tokenCount[bestStart[start]] + idCount(normalized, bestStart[start], bestId[start]);
                if (spaceStartsPiece && tokenCount[start] >= minTokens && normalized.charAt(start) == SPACE_SYMBOL) {
                    stop = start;
                    break;
                }
            }
            float scoreSoFar = bestScore[start];
            boolean hasSingleCharacter = false;
            int lastEnd = Math.min(length, start + maxPieceLength);
            for (int end = start + 1; end <= lastEnd; end++) {
                Integer id = normalPieces.get(normalized.substring(start, end));
                if (id == null) {
                    continue;
                }
                float candidate = scores[id] + scoreSoFar;
                if (bestStart[end] == -1 || candidate > bestScore[end]) {
                    bestScore[end] = candidate;
                    bestStart[end] = start;
                    bestId[end] = id;
                }
                if (end - start == charLength) {
                    hasSingleCharacter = true;
                }
            }
            if (!hasSingleCharacter) {
                int end = start + charLength;
                float candidate = unknownScore + scoreSoFar;
                if (bestStart[end] == -1 || candidate > bestScore[end]) {
                    bestScore[end] = candidate;
                    bestStart[end] = start;
                    bestId[end] = unknownId;
                }
            }
            start += charLength;
        }

        List<int[]> path = new ArrayList<>();
        for (int end = stop; end > 0; end = bestStart[end]) {
            path.add(new int[] {bestStart[end], end, bestId[end]});
        }
        int[] ids = new int[stop * 3];
        int count = 0;
        for (int i = path.size() - 1; i >= 0; i--) {
            int[] node = path.get(i);
            if (node[2] != unknownId) {
                ids[count++] = node[2];
                continue;
            }
            byte[] bytes = normalized.substring(node[0], node[1]).getBytes(StandardCharsets.UTF_8);
            for (byte b : bytes) {
                ids[count++] = byteIds[b & 0xFF];
            }
        }
        return Arrays.copyOf(ids, count);
    }

    /** Number of ids the piece starting at {@code start} produces, counting byte fallback. */
    private int idCount(String normalized, int start, int id) {
        if (id != unknownId) {
            return 1;
        }
        int end = start + Character.charCount(normalized.codePointAt(start));
        return normalized.substring(start, end).getBytes(StandardCharsets.UTF_8).length;
    }

    private static SentencePieceNormalizer readNormalizer(NormalizerSpec spec) {
        if (!spec.getNormalizationRuleTsv().isEmpty()) {
            throw new IllegalArgumentException("normalization_rule_tsv is not supported");
        }
        return new SentencePieceNormalizer(spec.getPrecompiledCharsmap().toByteArray(),
                spec.getAddDummyPrefix(), spec.getRemoveExtraWhitespaces(), spec.getEscapeWhitespaces());
    }

    private static int parseByte(String piece) {
        if (piece.length() != 6 || !piece.startsWith("<0x") || !piece.endsWith(">")) {
            throw new IllegalArgumentException("Malformed byte piece " + piece);
        }
        return Integer.parseInt(piece.substring(3, 5), 16);
    }
}
