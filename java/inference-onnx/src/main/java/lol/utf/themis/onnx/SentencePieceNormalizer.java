package lol.utf.themis.onnx;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;

/**
 * Port of SentencePiece's normalizer: applies a precompiled charsmap (a
 * darts-clone double-array trie over UTF-8 bytes) and its whitespace rules.
 * User-defined symbols are passed through unnormalized.
 */
final class SentencePieceNormalizer {

    private static final byte[] SPACE_SYMBOL = "\u2581".getBytes(StandardCharsets.UTF_8);
    private static final byte[] REPLACEMENT = "\uFFFD".getBytes(StandardCharsets.UTF_8);

    private final int[] trie;
    private final byte[] replacements;
    private final boolean addDummyPrefix;
    private final boolean removeExtraWhitespaces;
    private final boolean escapeWhitespaces;
    private final byte[][] userDefinedSymbols;
    /** Whether any user-defined symbol starts with a given byte. */
    private final boolean[] userDefinedFirstBytes = new boolean[256];

    /** An empty charsmap leaves characters unchanged. */
    SentencePieceNormalizer(byte[] precompiledCharsmap, boolean addDummyPrefix,
                            boolean removeExtraWhitespaces, boolean escapeWhitespaces,
                            List<String> userDefinedSymbols) {
        this.addDummyPrefix = addDummyPrefix;
        this.removeExtraWhitespaces = removeExtraWhitespaces;
        this.escapeWhitespaces = escapeWhitespaces;
        this.userDefinedSymbols = new byte[userDefinedSymbols.size()][];
        for (int i = 0; i < userDefinedSymbols.size(); i++) {
            byte[] symbol = userDefinedSymbols.get(i).getBytes(StandardCharsets.UTF_8);
            if (symbol.length == 0) {
                throw new IllegalArgumentException("User-defined symbols must not be empty");
            }
            this.userDefinedSymbols[i] = symbol;
            userDefinedFirstBytes[symbol[0] & 0xFF] = true;
        }
        if (precompiledCharsmap.length == 0) {
            trie = null;
            replacements = new byte[0];
            return;
        }
        ByteBuffer blob = ByteBuffer.wrap(precompiledCharsmap).order(ByteOrder.LITTLE_ENDIAN);
        if (blob.remaining() < 4) {
            throw new IllegalArgumentException("Truncated precompiled charsmap");
        }
        long trieSize = Integer.toUnsignedLong(blob.getInt());
        if (trieSize % 4 != 0 || trieSize > blob.remaining()) {
            throw new IllegalArgumentException("Malformed precompiled charsmap");
        }
        trie = new int[(int) trieSize / 4];
        blob.asIntBuffer().get(trie);
        replacements = Arrays.copyOfRange(precompiledCharsmap, 4 + (int) trieSize, precompiledCharsmap.length);
    }

    String normalize(String text) {
        byte[] input = text.getBytes(StandardCharsets.UTF_8);
        Output output = new Output(input.length * 3 / 2 + SPACE_SYMBOL.length);
        Prefix prefix = new Prefix();
        int position = 0;
        if (removeExtraWhitespaces) {
            while (position < input.length) {
                prefix.match(input, position);
                if (!prefix.isSingleSpace()) {
                    break;
                }
                position += prefix.consumed;
            }
        }
        if (position == input.length) {
            return "";
        }
        if (addDummyPrefix) {
            output.appendSpace();
        }
        boolean previousSpace = removeExtraWhitespaces;
        while (position < input.length) {
            prefix.match(input, position);
            int start = prefix.from;
            if (previousSpace) {
                while (start < prefix.to && prefix.source[start] == ' ') {
                    start++;
                }
            }
            if (start < prefix.to) {
                for (int i = start; i < prefix.to; i++) {
                    if (prefix.source[i] == ' ') {
                        output.appendSpace();
                    } else {
                        output.append(prefix.source[i]);
                    }
                }
                previousSpace = prefix.source[prefix.to - 1] == ' ';
            }
            position += prefix.consumed;
            if (!removeExtraWhitespaces) {
                previousSpace = false;
            }
        }
        if (removeExtraWhitespaces) {
            output.stripTrailingSpaces();
        }
        return new String(output.bytes, 0, output.size, StandardCharsets.UTF_8);
    }

    /**
     * The longest user-defined symbol at a position, unchanged, else the
     * replacement for the longest charsmap rule matching there.
     */
    private final class Prefix {
        byte[] source;
        int from;
        int to;
        int consumed;

        void match(byte[] input, int position) {
            int symbolLength = userDefinedSymbolLength(input, position);
            if (symbolLength > 0) {
                source = input;
                from = position;
                to = position + symbolLength;
                consumed = symbolLength;
                return;
            }
            int longestLength = 0;
            int longestValue = 0;
            if (trie != null) {
                int node = offset(trie[0]);
                for (int i = position; i < input.length; i++) {
                    int label = input[i] & 0xFF;
                    node ^= label;
                    if (node < 0 || node >= trie.length) {
                        break;
                    }
                    int unit = trie[node];
                    if ((unit & (0x80000000 | 0xFF)) != label) {
                        break;
                    }
                    node ^= offset(unit);
                    if (((unit >>> 8) & 1) == 1 && node < trie.length) {
                        longestLength = i - position + 1;
                        longestValue = trie[node] & 0x7FFFFFFF;
                    }
                }
            }
            if (longestLength > 0) {
                source = replacements;
                from = longestValue;
                to = from;
                while (to < replacements.length && replacements[to] != 0) {
                    to++;
                }
                consumed = longestLength;
                return;
            }
            int length = validUtf8Length(input, position);
            if (length == 0) {
                source = REPLACEMENT;
                from = 0;
                to = REPLACEMENT.length;
                consumed = 1;
            } else {
                source = input;
                from = position;
                to = position + length;
                consumed = length;
            }
        }

        boolean isSingleSpace() {
            return to - from == 1 && source[from] == ' ';
        }
    }

    private int userDefinedSymbolLength(byte[] input, int position) {
        int longest = 0;
        if (!userDefinedFirstBytes[input[position] & 0xFF]) {
            return longest;
        }
        for (byte[] symbol : userDefinedSymbols) {
            int end = position + symbol.length;
            if (symbol.length > longest && end <= input.length
                    && Arrays.equals(input, position, end, symbol, 0, symbol.length)) {
                longest = symbol.length;
            }
        }
        return longest;
    }

    private final class Output {
        byte[] bytes;
        int size;

        Output(int capacity) {
            bytes = new byte[Math.max(capacity, 16)];
        }

        void append(byte b) {
            if (size == bytes.length) {
                bytes = Arrays.copyOf(bytes, bytes.length * 2);
            }
            bytes[size++] = b;
        }

        void appendSpace() {
            if (escapeWhitespaces) {
                for (byte b : SPACE_SYMBOL) {
                    append(b);
                }
            } else {
                append((byte) ' ');
            }
        }

        void stripTrailingSpaces() {
            byte[] space = escapeWhitespaces ? SPACE_SYMBOL : new byte[] {' '};
            while (size >= space.length
                    && Arrays.equals(bytes, size - space.length, size, space, 0, space.length)) {
                size -= space.length;
            }
        }
    }

    private static int offset(int unit) {
        return (unit >>> 10) << ((unit & (1 << 9)) >>> 6);
    }

    /** Returns the length of the UTF-8 character at position, or 0 if malformed. */
    private static int validUtf8Length(byte[] input, int position) {
        int lead = input[position] & 0xFF;
        int length;
        int codePoint;
        if (lead < 0x80) {
            return 1;
        } else if (lead >= 0xC2 && lead <= 0xDF) {
            length = 2;
            codePoint = lead & 0x1F;
        } else if (lead >= 0xE0 && lead <= 0xEF) {
            length = 3;
            codePoint = lead & 0x0F;
        } else if (lead >= 0xF0 && lead <= 0xF4) {
            length = 4;
            codePoint = lead & 0x07;
        } else {
            return 0;
        }
        if (position + length > input.length) {
            return 0;
        }
        for (int i = 1; i < length; i++) {
            int b = input[position + i] & 0xFF;
            if ((b & 0xC0) != 0x80) {
                return 0;
            }
            codePoint = (codePoint << 6) | (b & 0x3F);
        }
        boolean overlong = (length == 3 && codePoint < 0x800) || (length == 4 && codePoint < 0x10000);
        boolean invalid = codePoint > 0x10FFFF || (codePoint >= 0xD800 && codePoint <= 0xDFFF);
        return overlong || invalid ? 0 : length;
    }
}
