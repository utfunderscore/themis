package lol.utf.themis.onnx;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/** Behaviour of the tokenizer on a small hand-built vocabulary. */
class TokenizerTest {

    private static final int CLS = 1;
    private static final int SEP = 2;

    private static final List<String> NORMAL = List.of("▁hello", "▁he", "llo", "▁", "h", "e", "l", "o", "▁world");
    private static final float[] SCORES = {-1, -2, -2, -3, -5, -5, -5, -5, -1};
    private static final int FIRST_NORMAL = 4 + 256;

    private static final SentencePiece MODEL = model(true);

    private static SentencePiece model(boolean addDummyPrefix) {
        List<SentencePiece.Piece> pieces = new ArrayList<>();
        pieces.add(new SentencePiece.Piece("[PAD]", 0, SentencePiece.CONTROL));
        pieces.add(new SentencePiece.Piece("[CLS]", 0, SentencePiece.CONTROL));
        pieces.add(new SentencePiece.Piece("[SEP]", 0, SentencePiece.CONTROL));
        pieces.add(new SentencePiece.Piece("[UNK]", 0, SentencePiece.UNKNOWN));
        for (int b = 0; b < 256; b++) {
            pieces.add(new SentencePiece.Piece(String.format("<0x%02X>", b), 0, SentencePiece.BYTE));
        }
        for (int i = 0; i < NORMAL.size(); i++) {
            pieces.add(new SentencePiece.Piece(NORMAL.get(i), SCORES[i], SentencePiece.NORMAL));
        }
        return new SentencePiece(pieces, new SentencePieceNormalizer(new byte[0], addDummyPrefix, true, true));
    }

    private static int piece(String piece) {
        return FIRST_NORMAL + NORMAL.indexOf(piece);
    }

    private static int bytePiece(int b) {
        return 4 + b;
    }

    @Test
    void choosesHighestScoringSegmentation() {
        assertArrayEquals(new int[] {piece("▁hello")}, MODEL.encode("hello"));
        assertArrayEquals(new int[] {piece("▁he"), piece("l"), piece("l")}, MODEL.encode("hell"));
    }

    @Test
    void collapsesAndTrimsWhitespace() {
        assertArrayEquals(new int[] {piece("▁hello"), piece("▁world")}, MODEL.encode("  hello    world  "));
        assertArrayEquals(new int[0], MODEL.encode("   "));
        assertArrayEquals(new int[0], MODEL.encode(""));
    }

    @Test
    void fallsBackToUtf8BytesForUnknownCharacters() {
        assertArrayEquals(new int[] {piece("▁he"), piece("l"), piece("l"), bytePiece(0xC3), bytePiece(0xB8)},
                MODEL.encode("hellø"));
        assertArrayEquals(new int[] {piece("▁"), bytePiece(0xF0), bytePiece(0x9F), bytePiece(0x91),
                bytePiece(0x8B)}, MODEL.encode("👋"));
    }

    @Test
    void omitsDummyPrefixWhenDisabled() {
        assertArrayEquals(new int[] {piece("h"), piece("e"), piece("llo")}, model(false).encode("hello"));
    }

    @Test
    void keepsSpecialTokensWholeAndWrapsSequence() {
        DebertaTokenizer tokenizer = new DebertaTokenizer(MODEL, Map.of("[CLS]", CLS, "[SEP]", SEP), CLS, SEP, 16);
        assertArrayEquals(new long[] {CLS, piece("▁hello"), SEP, piece("▁world"), SEP},
                tokenizer.encode("hello[SEP] world").inputIds());
        assertArrayEquals(new long[] {CLS, SEP}, tokenizer.encode("").inputIds());
        assertArrayEquals(new long[] {CLS, CLS, SEP}, tokenizer.encode("[CLS]").inputIds());
    }

    @Test
    void truncatesContentToMaxLength() {
        DebertaTokenizer tokenizer = new DebertaTokenizer(MODEL, Map.of("[SEP]", SEP), CLS, SEP, 4);
        assertArrayEquals(new long[] {CLS, piece("▁hello"), piece("▁world"), SEP},
                tokenizer.encode("hello world hello").inputIds());
        assertArrayEquals(new long[] {CLS, piece("▁hello"), SEP, SEP},
                tokenizer.encode("hello[SEP]world").inputIds());
        assertThrows(IllegalArgumentException.class, () -> new DebertaTokenizer(MODEL, Map.of(), CLS, SEP, 1));
    }
}
