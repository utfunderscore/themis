# Java inference

Java 21 Gradle project for implementing inference with the exported
DeBERTa ONNX model.

## Module

- `inference-onnx`: inference implementation using the CPU ONNX Runtime Java library.

This is an initial build scaffold. Model loading, tokenization, inference, and
label decoding are still to be implemented. Source packages use `lol.utf.themis`.
`inference-onnx` is a Gradle subproject, not a JPMS module.

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
Add Java tests under each module's `src/test/java/` and configure a test framework
when tests are introduced.

## Model integration

The Python exporter is `src/themis/models/deberta/onnx_export.py` relative
to the repository root. Use its exported model bundle when implementing the
Java loader, keeping tokenizer behavior and label order consistent with Python.
Model bundles should stay external to the library JARs.
