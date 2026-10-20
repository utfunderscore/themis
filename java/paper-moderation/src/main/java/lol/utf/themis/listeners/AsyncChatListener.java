package lol.utf.themis.listeners;

import io.papermc.paper.event.player.AsyncChatEvent;
import lol.utf.themis.ThemisPlugin;
import net.kyori.adventure.text.logger.slf4j.ComponentLogger;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.jetbrains.annotations.NotNull;
import org.jspecify.annotations.NonNull;
import org.slf4j.Logger;

import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class AsyncChatListener implements Listener {

    @NotNull private final ComponentLogger logger;

    private final Executor executor = Executors.newVirtualThreadPerTaskExecutor();

    public AsyncChatListener(@NotNull ComponentLogger logger) {
        this.logger = logger;
    }

    @EventHandler
    public void onChat(@NonNull AsyncChatEvent event) {
        if (event.isAsynchronous()) {



            return;
        }




    }

}
