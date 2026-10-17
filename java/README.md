# Java inference

This Java 21 project lets you use the Themis text moderation model from a Java
application. It contains a library (`inference-onnx`) and a small command-line
tool (`inference-cli`). Both use an exported model bundle produced by the
Python project; the bundle is a directory on disk, not part of the Java JAR.

## Use the moderation API

Load a model bundle once, use the `Moderator` to score text, then close it when
you are done. The `try` block below closes the model automatically:

```java
import java.nio.file.Path;
import lol.utf.themis.onnx.Action;
import lol.utf.themis.onnx.Category;
import lol.utf.themis.onnx.Moderation;
import lol.utf.themis.onnx.Moderator;

Path bundle = Path.of("/path/to/exported/model-bundle");

try (Moderator moderator = Moderator.load(bundle)) {
    Moderation result = moderator.moderate("hello world");

    System.out.println("Category: " + result.category());
    System.out.println("Action: " + result.action());
    System.out.println("Insult probability: " + result.probability(Category.INSULT));

    if (result.action() == Action.BLOCK_AND_REVIEW) {
        System.out.println("Send this text for review.");
    }
}
```

`category()` and `action()` return the most likely labels. `probability(...)`
returns the probability for a particular label. Categories are `NONE`,
`PROFANITY`, `INSULT`, `SLUR`, `SEXUAL`, and `THREAT`; actions are `ALLOW` and
`BLOCK_AND_REVIEW`. The action and category are scored separately.

For several texts, use `moderateAll(List<String>)`; results come back in the
same order as the input. If your application uses asynchronous code, `submit`
and `submitAll` return `CompletableFuture` results instead of waiting for
inference to finish.

## Run the CLI

The CLI reads one text per line from standard input and prints the action,
category, category probability, and original text for each line. Provide the
path to an exported model bundle as its argument:

```sh
./gradlew :inference-cli:run --args="/path/to/exported/model-bundle"
```

Type or paste text, one line at a time. Press Ctrl-D on macOS/Linux or Ctrl-Z
then Enter on Windows when you are finished. You can also pipe in a file:

```sh
./gradlew :inference-cli:run --args="/path/to/exported/model-bundle" < input.txt
```

To create a standalone CLI distribution, run `./gradlew
:inference-cli:installDist`. Then run the generated `inference-cli` script in
`inference-cli/build/install/inference-cli/bin/`, passing the bundle path in
the same way.

## Build the project

Install a JDK 21 and set `JAVA_HOME` to its installation directory. In a
terminal, go to this `java` directory and run:

```sh
./gradlew build
```

On Windows, run `gradlew.bat build`. The first build downloads Gradle and the
project's dependencies. To build only the library, use
`./gradlew :inference-onnx:build`; to build only the CLI, use
`./gradlew :inference-cli:build`.

The library is provided as a Gradle project dependency. In a Gradle project
that includes this source tree, add:

```kotlin
dependencies {
    implementation(project(":inference-onnx"))
}
```

If you are using the library from a separate project, its JAR is created in
`inference-onnx/build/libs/`. The ONNX Runtime, Gson, and Protobuf Java dependencies are also
needed at runtime, so make sure your application resolves the library's
dependencies as well.
