# Themis moderation (Java)

This library checks a piece of text and tells you whether it contains something
like profanity, insults or threats, and whether it should be held for a human
to review.

## Download a model automatically

`ModelDownloader` fetches the official model from Hugging Face
([`utfunderscore/themis-moderation`](https://huggingface.co/utfunderscore/themis-moderation)):

```java
import lol.utf.themis.onnx.ModelDownloader;
import lol.utf.themis.onnx.Moderator;

try (Moderator moderator = ModelDownloader.download()
        .thenCompose(Moderator::load).join()) {
    System.out.println(moderator.moderate("hello world").join().action());
}
```

The first call downloads `model.onnx`, `spm.model`, `tokenizer_config.json`, and
`inference_config.json` from the repository's `onnx/` folder into a `themis`
folder in the working directory (where the `java` command was started). Later
calls reuse the cached bundle without network access.
To choose a cache location, use `ModelDownloader.download(Path.of("/your/cache"))`.

Each library version is pinned to one model commit
(`ModelDownloader.DEFAULT_REVISION`), so upgrading the library is how you get a
newer model. To use a different repository or revision, call
`ModelDownloader.download("OWNER/MODEL", "COMMIT")` with a public repository
published by `themis.models.deberta.hub.upload_model`. Use a commit ID: branch
and tag names work, but cached downloads are not checked for updates.

Downloads stream to disk and are published to the cache only after every file
succeeds. Failures complete the future exceptionally; calling again retries a
failed download. Private and gated repositories are not supported. Model loading
performs compatibility validation after downloading.

## Using it

If you already have a **model folder** (made by the Python exporter in this
repo), load it directly, then give it text:

```java
import java.nio.file.Path;
import lol.utf.themis.onnx.*;

try (Moderator moderator = Moderator.load(Path.of("/path/to/model-folder")).join()) {

    Moderation result = moderator.moderate("hello world").join();

    System.out.println(result.category()); // NONE
    System.out.println(result.action());   // ALLOW
}
```

Downloading, loading and moderating return a
`CompletableFuture`, which is Java's way of saying "the answer will arrive
later". Any problem (a missing model folder, a closed moderator, a failing
model) comes out through that future instead. The examples use `.join()` to wait
for the answer; see "When things go wrong" for what it throws.

The `try (...)` part makes sure the model is shut down properly when you're done.

## Reading the result

A result has two answers:

- **`category()`**: what kind of text it is. One of `NONE`, `PROFANITY`,
  `INSULT`, `SLUR`, `SEXUAL`, `THREAT`.
- **`action()`**: what to do with it. Either `ALLOW` or `BLOCK_AND_REVIEW`
  (hold the message and let a person look at it).

These are decided separately, so a message can be `NONE` and still be
`BLOCK_AND_REVIEW`.

Want to know how confident the model is? Ask for a probability between 0
(no chance) and 1 (certain):

```java
double insultChance = result.probability(Category.INSULT);
double reviewChance = result.probability(Action.BLOCK_AND_REVIEW);
```

Typical use:

```java
if (result.action() == Action.BLOCK_AND_REVIEW) {
    // don't show the message; send it to a moderator
}
```

## Checking many messages

```java
List<Moderation> results = moderator.moderateAll(List.of("first", "second", "third")).join();
```

The results come back in the same order as the messages you gave.

## Not waiting for the answer

Instead of `.join()`, attach what should happen next:

```java
moderator.moderate("some text").thenAccept(result -> {
    System.out.println(result.action());
});
```

Keep that callback quick, and don't `.join()` on another request inside it. If
you need to close the moderator from a callback, use `closeAsync()`, which
returns a future and doesn't wait.

It's safe to call the same `Moderator` from many threads at once. Create one
and share it rather than making a new one each time (loading is slow).

## Speeding it up

By default each message is checked one at a time. If lots of messages arrive at
once, let the library check several together:

```java
Moderator moderator = Moderator.load(Path.of("/path/to/model-folder"), 16).join();
```

`16` is the most messages it will check together. It never waits around to fill
a group, so this won't slow down quiet periods.

## When things go wrong

Failures arrive through the future. `.join()` throws a `CompletionException`
whose cause is the real problem, `.get()` throws an `ExecutionException`, and
`.exceptionally(...)` / `.handle(...)` let you deal with it without throwing:

```java
moderator.moderate(text).exceptionally(failure -> { log(failure); return null; });
```

The cause is one of:

- **`InferenceException`**: the model folder is missing or wrong, or the model
  failed while running.
- **`RejectedExecutionException`**: you used the moderator after closing it.
- **`NullPointerException` / `IllegalArgumentException`**: a null text, or a
  batch size below 1.

Separately, a **"waiting requests" warning in your logs** means messages
are arriving faster than the model can check them (more than 1024 are waiting by
default). Nothing is dropped, but results will lag and memory use grows until
the backlog clears. Try a larger batch size, or change the threshold with the
`THEMIS_QUEUE_WARNING_THRESHOLD` environment variable.
