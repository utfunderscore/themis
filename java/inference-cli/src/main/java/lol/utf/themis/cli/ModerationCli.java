package lol.utf.themis.cli;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Locale;
import lol.utf.themis.onnx.Action;
import lol.utf.themis.onnx.Category;
import lol.utf.themis.onnx.Moderation;
import lol.utf.themis.onnx.Moderator;
import lol.utf.themis.onnx.Scheduling;

/**
 * Reads one text per line and moderates each with the ONNX model in FIFO order,
 * printing one result line per input. Blank lines are skipped.
 *
 * <p>Usage: {@code ModerationCli <bundle-directory>}, with texts on standard input.
 * Output is {@code action<TAB>category<TAB>category-probability<TAB>text}.
 */
public final class ModerationCli {

    private ModerationCli() {
    }

    public static void main(String[] args) throws IOException {
        if (args.length != 1) {
            System.err.println("Usage: ModerationCli <bundle-directory>   (texts on stdin, one per line)");
            System.exit(2);
        }
        System.exit(run(Path.of(args[0]), System.in, System.out, System.err));
    }

    /**
     * Moderates every non-blank line of {@code in}, in order.
     *
     * @return the process exit code: 0 on success, 1 if any line failed
     */
    static int run(Path bundle, InputStream in, PrintStream out, PrintStream err) throws IOException {
        int failures = 0;
        try (Moderator moderator = Moderator.load(bundle, Scheduling.fifo());
             BufferedReader reader = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isBlank()) {
                    continue;
                }
                try {
                    out.println(format(moderator.moderate(line), line));
                } catch (RuntimeException e) {
                    failures++;
                    err.println("error: " + e.getMessage() + " (input: " + line + ")");
                }
                out.flush();
            }
        }
        return failures == 0 ? 0 : 1;
    }

    static String format(Moderation result, String text) {
        Action action = result.action();
        Category category = result.category();
        return String.format(Locale.ROOT, "%s\t%s\t%.4f\t%s",
                action.label(), category.label(), result.probability(category), text);
    }
}
