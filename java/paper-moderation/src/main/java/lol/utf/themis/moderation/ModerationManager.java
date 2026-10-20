package lol.utf.themis.moderation;

import lol.utf.themis.onnx.ModelDownloader;
import lol.utf.themis.onnx.Moderator;
import net.kyori.adventure.text.Component;
import org.jspecify.annotations.NonNull;

import java.nio.file.Path;

public class ModerationManager {

    private final Moderator moderator;

    public ModerationManager(@NonNull Path pluginDir) {
        Path modelPath = pluginDir.resolve("model");
        Path cacheDirectory = ModelDownloader.download(modelPath).join();
        this.moderator = Moderator.load(cacheDirectory, 1).join();
    }

    public void moderate(Component chatComponent) {
    }

}
