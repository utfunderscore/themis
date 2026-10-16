# Java inference

Java 21 Gradle project for implementing inference with the exported
DeBERTa ONNX model.

## Module

- `inference-onnx`: inference implementation using the CPU ONNX Runtime Java library.

Source packages use `lol.utf.themis`. `inference-onnx` is a Gradle subproject,
not a JPMS module.

## Moderation API

`Moderator` loads a bundle directory exported by the Python exporter and turns
text into a `Moderation`, matching Python's `Moderation.predict`: the most
probable `Category` (`NONE`, `PROFANITY`, `INSULT`, `SLUR`, `SEXUAL`, `THREAT`)
and `Action` (`ALLOW`, `BLOCK_AND_REVIEW`), plus each group's independent
probabilities. `BLOCK_AND_REVIEW` means send for human review, independently
of the category.

```java
try (Moderator moderator = Moderator.load(bundle, Scheduling.batched(16))) {
    Moderation result = moderator.moderate("hello world");
    double insult = result.probability(Category.INSULT);
    if (result.action() == Action.BLOCK_AND_REVIEW) { ... }
}
```

- `moderate(text)` / `moderateAll(texts)` block; `submit(text)` /
  `submitAll(texts)` return `CompletableFuture`s. All are thread safe.
- Requests from all threads share one FIFO queue served by a single inference
  thread, so the model never runs concurrently. The thread starts on a request
  as soon as it is free; it never waits for a batch to fill.
- `Scheduling.fifo()` (the default) runs one text per model call.
  `Scheduling.batched(maxBatchSize)` lets texts that queued while a call was
  running share the next call. `withQueueCapacity` bounds the queue; a full
  queue throws `RejectedExecutionException`.
- Futures complete on the inference thread, so use `*Async` stages for slow
  follow-up work. Blocking calls made from that thread throw
  `IllegalStateException` instead of deadlocking.
- `close()` finishes queued requests, then releases the model.

ONNX Runtime still parallelizes each model call across CPU cores. To keep
inference on one core, pass session options with `setIntraOpNumThreads(1)` to
`Moderator.load(bundle, scheduling, sessionOptions)`.

Tokenization is a pure-Java port of the slow `DebertaV2Tokenizer` (SentencePiece
unigram with `nmt_nfkc` normalization and byte fallback). Texts are truncated to
the bundle's `max_length`. Loading validates `inference_config.json` against the
labels, tokenizer settings and postprocessing implemented here.

## Build

Install a JDK 21 and set `JAVA_HOME` to its installation directory. From this
directory, run:

```sh
./gradlew build
```

On Windows use `gradlew.bat build`. The wrapper downloads the pinned Gradle
distribution on first use; dependencies are resolved from Maven Central.
Compilation uses the Java 21 toolchain and release target.

Module JARs and source JARs are written to each module's `build/libs/` directory.
These are library JARs; dependencies must also be on the consuming application's
runtime classpath.

Dependency versions are maintained in `gradle/libs.versions.toml`.
Tests use JUnit and run with `./gradlew build`. `ModeratorTest` and `OnnxModelTest`
compare tokenization, logits and probabilities with Python outputs and are skipped
unless an exported bundle is given:

```sh
./gradlew test -Pthemis.onnx.bundle=/path/to/bundle
```

## Model integration

The Python exporter is `src/themis/models/deberta/onnx_export.py` relative
to the repository root. If tokenizer behavior or label order changes in Python,
update the Java tokenizer and `Category`/`Action` enums to match. Model bundles
should stay external to the library JARs.
