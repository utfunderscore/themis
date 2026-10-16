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

    private static final int UNIGRAM = 1;
    private static final float UNKNOWN_PENALTY = 10.0f;

    record Piece(String piece, float score, int type) {
    }

    private final SentencePieceNormalizer normalizer;
    private final Map<String, Integer> normalPieces = new HashMap<>();
    private final float[] scores;
    private final int[] byteIds = new int[256];
    private final int unknownId;
    private final float unknownScore;
    private final int maxPieceLength;

    SentencePiece(List<Piece> pieces, SentencePieceNormalizer normalizer) {
        this.normalizer = normalizer;
        this.scores = new float[pieces.size()];
        Arrays.fill(byteIds, -1);
        int unknown = -1;
        float minScore = Float.MAX_VALUE;
        int longest = 0;
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
    }

    /** Loads a unigram {@code .model} file trained with byte fallback. */
    static SentencePiece load(Path modelFile) throws IOException {
        ProtobufReader model = new ProtobufReader(Files.readAllBytes(modelFile));
        List<Piece> pieces = new ArrayList<>();
        SentencePieceNormalizer normalizer = null;
        int modelType = UNIGRAM;
        boolean byteFallback = false;
        boolean whitespaceSuffix = false;
        while (model.next()) {
            switch (model.field()) {
                case 1 -> pieces.add(readPiece(model.readMessage()));
                case 2 -> {
                    ProtobufReader trainer = model.readMessage();
                    while (trainer.next()) {
                        switch (trainer.field()) {
                            case 3 -> modelType = (int) trainer.readVarint();
                            case 24 -> whitespaceSuffix = trainer.readBool();
                            case 35 -> byteFallback = trainer.readBool();
                            default -> trainer.skip();
                        }
                    }
                }
                case 3 -> normalizer = readNormalizer(model.readMessage());
                default -> model.skip();
            }
        }
        if (modelType != UNIGRAM) {
            throw new IllegalArgumentException("Only unigram SentencePiece models are supported");
        }
        if (!byteFallback) {
            throw new IllegalArgumentException("Only SentencePiece models with byte fallback are supported");
        }
        if (whitespaceSuffix) {
            throw new IllegalArgumentException("treat_whitespace_as_suffix is not supported");
        }
        if (normalizer == null) {
            normalizer = new SentencePieceNormalizer(new byte[0], true, true, true);
        }
        return new SentencePiece(pieces, normalizer);
    }

    /** Encodes text to piece ids, without special tokens. */
    int[] encode(String text) {
        String normalized = normalizer.normalize(text);
        int length = normalized.length();
        if (length == 0) {
            return new int[0];
        }
        float[] bestScore = new float[length + 1];
        int[] bestStart = new int[length + 1];
        int[] bestId = new int[length + 1];
        Arrays.fill(bestStart, -1);

        for (int start = 0; start < length; ) {
            int charLength = Character.charCount(normalized.codePointAt(start));
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
        for (int end = length; end > 0; end = bestStart[end]) {
            path.add(new int[] {bestStart[end], end, bestId[end]});
        }
        int[] ids = new int[length * 3];
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

    private static Piece readPiece(ProtobufReader message) {
        String piece = "";
        float score = 0;
        int type = NORMAL;
        while (message.next()) {
            switch (message.field()) {
                case 1 -> piece = message.readString();
                case 2 -> score = message.readFloat();
                case 3 -> type = (int) message.readVarint();
                default -> message.skip();
            }
        }
        return new Piece(piece, score, type);
    }

    private static SentencePieceNormalizer readNormalizer(ProtobufReader message) {
        byte[] charsmap = new byte[0];
        boolean addDummyPrefix = true;
        boolean removeExtraWhitespaces = true;
        boolean escapeWhitespaces = true;
        while (message.next()) {
            switch (message.field()) {
                case 2 -> charsmap = message.readBytes();
                case 3 -> addDummyPrefix = message.readBool();
                case 4 -> removeExtraWhitespaces = message.readBool();
                case 5 -> escapeWhitespaces = message.readBool();
                case 6 -> {
                    if (!message.readString().isEmpty()) {
                        throw new IllegalArgumentException("normalization_rule_tsv is not supported");
                    }
                }
                default -> message.skip();
            }
        }
        return new SentencePieceNormalizer(charsmap, addDummyPrefix, removeExtraWhitespaces, escapeWhitespaces);
    }

    private static int parseByte(String piece) {
        if (piece.length() != 6 || !piece.startsWith("<0x") || !piece.endsWith(">")) {
            throw new IllegalArgumentException("Malformed byte piece " + piece);
        }
        return Integer.parseInt(piece.substring(3, 5), 16);
    }
}
