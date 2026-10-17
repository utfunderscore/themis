package lol.utf.themis.onnx;

import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.UnknownFieldSet;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import sentencepiece.SentencepieceModel.ModelProto;
import sentencepiece.SentencepieceModel.NormalizerSpec;
import sentencepiece.SentencepieceModel.TrainerSpec;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class SentencePieceTest {
    @TempDir Path directory;

    @Test
    void schemaDefaultsNormalizeWhitespaceAndDefaultPieceType() throws IOException {
        // Omitted model type defaults to unigram; omitted piece type/score to NORMAL/0.
        ModelProto.Builder model = validModel();
        model.addPieces(ModelProto.SentencePiece.newBuilder().setPiece("▁a"));
        assertArrayEquals(new int[] {257}, load(model).encode("  a  "));
        // An explicit empty normalizer has the same defaults as an absent one.
        model.setNormalizerSpec(NormalizerSpec.getDefaultInstance());
        assertArrayEquals(new int[] {257}, load(model).encode("  a  "));
    }

    @Test
    void explicitNormalizerFlagsAndByteFallbackArePreserved() throws IOException {
        ModelProto.Builder model = validModel().setNormalizerSpec(NormalizerSpec.newBuilder()
                .setAddDummyPrefix(false).setRemoveExtraWhitespaces(false).setEscapeWhitespaces(false));
        model.addPieces(ModelProto.SentencePiece.newBuilder().setPiece("a"));
        assertArrayEquals(new int[] {33, 257, 33}, load(model).encode(" a "));
        assertArrayEquals(new int[] {196, 170}, load(model).encode("é"));
    }

    @Test
    void unsupportedModelsAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> load(validModel().clearTrainerSpec()));
        assertThrows(IllegalArgumentException.class, () -> load(validModel().setTrainerSpec(
                TrainerSpec.newBuilder().setByteFallback(true).setModelType(TrainerSpec.ModelType.BPE))));
        assertThrows(IllegalArgumentException.class, () -> load(validModel().setTrainerSpec(
                TrainerSpec.newBuilder().setByteFallback(true).setTreatWhitespaceAsSuffix(true))));
        assertThrows(IllegalArgumentException.class, () -> load(validModel().setNormalizerSpec(
                NormalizerSpec.newBuilder().setNormalizationRuleTsv("rules.tsv"))));
        assertThrows(IllegalArgumentException.class, () -> load(validModel().removePieces(1)));
    }

    @Test
    void unknownEnumValuesAreRejectedInsteadOfTakingSchemaDefaults() {
        assertThrows(IllegalArgumentException.class, () -> load(validModel().setTrainerSpec(
                TrainerSpec.newBuilder().setByteFallback(true).setUnknownFields(
                        unknownVarint(TrainerSpec.MODEL_TYPE_FIELD_NUMBER, 99)))));
        assertThrows(IllegalArgumentException.class, () -> load(validModel().addPieces(
                ModelProto.SentencePiece.newBuilder().setPiece("x").setUnknownFields(
                        unknownVarint(ModelProto.SentencePiece.TYPE_FIELD_NUMBER, 99)))));
    }

    @Test
    void unrelatedUnknownFieldsRemainForwardCompatible() throws IOException {
        ModelProto.Builder model = validModel().setUnknownFields(unknownVarint(1000, 42));
        model.addPieces(ModelProto.SentencePiece.newBuilder().setPiece("▁a"));
        assertArrayEquals(new int[] {257}, load(model).encode("a"));
    }

    @Test
    void malformedWireDataThrowsIOException() throws IOException {
        for (byte[] bytes : new byte[][] {{0}, {10, 5, 1}, {18, (byte) 128}, {(byte) 128}}) {
            Path path = directory.resolve("broken.model");
            Files.write(path, bytes);
            assertThrows(InvalidProtocolBufferException.class, () -> SentencePiece.load(path));
        }
    }

    private SentencePiece load(ModelProto.Builder model) throws IOException {
        Path path = directory.resolve("test.model");
        Files.write(path, model.build().toByteArray());
        return SentencePiece.load(path);
    }

    private static UnknownFieldSet unknownVarint(int field, long value) {
        return UnknownFieldSet.newBuilder().addField(field,
                UnknownFieldSet.Field.newBuilder().addVarint(value).build()).build();
    }

    private static ModelProto.Builder validModel() {
        ModelProto.Builder model = ModelProto.newBuilder().setTrainerSpec(
                TrainerSpec.newBuilder().setByteFallback(true));
        model.addPieces(ModelProto.SentencePiece.newBuilder().setPiece("<unk>")
                .setType(ModelProto.SentencePiece.Type.UNKNOWN));
        for (int b = 0; b < 256; b++) {
            model.addPieces(ModelProto.SentencePiece.newBuilder().setPiece("<0x%02X>".formatted(b))
                    .setType(ModelProto.SentencePiece.Type.BYTE));
        }
        return model;
    }
}
