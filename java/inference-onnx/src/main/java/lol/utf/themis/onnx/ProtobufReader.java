package lol.utf.themis.onnx;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;

/** Minimal protobuf wire-format reader, enough to parse SentencePiece models. */
final class ProtobufReader {

    static final int VARINT = 0;
    static final int FIXED64 = 1;
    static final int LENGTH_DELIMITED = 2;
    static final int FIXED32 = 5;

    private final byte[] data;
    private int position;
    private final int limit;
    private int tag;

    ProtobufReader(byte[] data) {
        this(data, 0, data.length);
    }

    private ProtobufReader(byte[] data, int offset, int limit) {
        this.data = data;
        this.position = offset;
        this.limit = limit;
    }

    /** Advances to the next field; returns false at the end of the message. */
    boolean next() {
        if (position >= limit) {
            return false;
        }
        tag = (int) rawVarint();
        return true;
    }

    int field() {
        return tag >>> 3;
    }

    int wireType() {
        return tag & 7;
    }

    long readVarint() {
        expect(VARINT);
        return rawVarint();
    }

    private long rawVarint() {
        long result = 0;
        for (int shift = 0; shift < 64; shift += 7) {
            byte b = readByte();
            result |= (long) (b & 0x7F) << shift;
            if (b >= 0) {
                return result;
            }
        }
        throw new IllegalArgumentException("Malformed varint");
    }

    boolean readBool() {
        return readVarint() != 0;
    }

    float readFloat() {
        expect(FIXED32);
        int bits = 0;
        for (int i = 0; i < 4; i++) {
            bits |= (readByte() & 0xFF) << (8 * i);
        }
        return Float.intBitsToFloat(bits);
    }

    byte[] readBytes() {
        int length = readLength();
        byte[] bytes = Arrays.copyOfRange(data, position, position + length);
        position += length;
        return bytes;
    }

    String readString() {
        return new String(readBytes(), StandardCharsets.UTF_8);
    }

    ProtobufReader readMessage() {
        int length = readLength();
        ProtobufReader message = new ProtobufReader(data, position, position + length);
        position += length;
        return message;
    }

    void skip() {
        switch (wireType()) {
            case VARINT -> readVarint();
            case FIXED64 -> advance(8);
            case LENGTH_DELIMITED -> advance(readLength());
            case FIXED32 -> advance(4);
            default -> throw new IllegalArgumentException("Unsupported wire type " + wireType());
        }
    }

    private int readLength() {
        expect(LENGTH_DELIMITED);
        long length = rawVarint();
        if (length < 0 || length > limit - position) {
            throw new IllegalArgumentException("Malformed length " + length);
        }
        return (int) length;
    }

    private void advance(int count) {
        if (count > limit - position) {
            throw new IllegalArgumentException("Truncated message");
        }
        position += count;
    }

    private byte readByte() {
        if (position >= limit) {
            throw new IllegalArgumentException("Truncated message");
        }
        return data[position++];
    }

    private void expect(int wireType) {
        if (wireType() != wireType) {
            throw new IllegalArgumentException(
                    "Field " + field() + " has wire type " + wireType() + ", expected " + wireType);
        }
    }
}
