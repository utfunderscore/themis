SentencePiece schema provenance

`sentencepiece_model.proto` is copied unchanged from Google SentencePiece v0.2.1,
commit `31646a467d2051eb904e0b45de3a73e91fe1c1e3`:
https://github.com/google/sentencepiece/blob/31646a467d2051eb904e0b45de3a73e91fe1c1e3/src/sentencepiece_model.proto

SHA-256: `798fb346a2934cc58a0f905be120f6710d0585847edf5425b7a1cf68102979bc`.
The original copyright header and Apache 2.0 license (`LICENSE.sentencepiece`)
are retained.

Gradle generates Java bindings during the build using the pinned protobuf plugin
and matching protoc/protobuf-java versions in `gradle/libs.versions.toml`.
The standard Java runtime is used; no generated sources need to be checked in.
The upstream `optimize_for = LITE_RUNTIME` option does not select Java lite output
with modern protoc; Java lite generation requires an explicit generator option.

Parsing uses the schema's proto2 defaults: unigram model, disabled byte fallback,
normal piece type, zero score, and all three normalization flags enabled.
Unknown enum values are explicitly rejected because proto2 otherwise exposes the
default value through its generated enum getter.
