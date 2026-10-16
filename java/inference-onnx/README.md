# Themis moderation (Java)

This library checks a piece of text and tells you whether it contains something
like profanity, insults or threats, and whether it should be held for a human
to review.

## Using it

You need a **model folder** (made by the Python exporter in this repo). Point
the library at it, then give it text:

```java
import java.nio.file.Path;
import lol.utf.themis.onnx.*;

try (Moderator moderator = Moderator.load(Path.of("/path/to/model-folder"))) {

    Moderation result = moderator.moderate("hello world");

    System.out.println(result.category()); // NONE
    System.out.println(result.action());   // ALLOW
}
```

That's all you need for most uses. The `try (...)` part makes sure the model is
shut down properly when you're done.

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
List<Moderation> results = moderator.moderateAll(List.of("first", "second", "third"));
```

The results come back in the same order as the messages you gave.

## If you're using several threads

It's safe to call the same `Moderator` from many threads at once. Create one
and share it rather than making a new one each time (loading is slow).

If you don't want your code to wait for the answer, use `submit` instead of
`moderate`. It returns right away with a `CompletableFuture`, which is Java's
way of saying "the answer will arrive later":

```java
moderator.submit("some text").thenAccept(result -> {
    System.out.println(result.action());
});
```

(Inside that callback, keep the work quick, and don't call `moderate` from
it.)

## Speeding it up

By default each message is checked one at a time. If lots of messages arrive at
once, let the library check several together:

```java
Moderator moderator = Moderator.load(Path.of("/path/to/model-folder"), Scheduling.batched(16));
```

`16` is the most messages it will check together. It never waits around to fill
a group, so this won't slow down quiet periods.

## When things go wrong

- **`InferenceException`**: the model folder is missing or wrong, or the model
  failed while running.
- **`RejectedExecutionException`**: you used the moderator after closing it, or
  too many messages are waiting (the default limit is 1024).
