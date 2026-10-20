package lol.utf.themis;

import lol.utf.themis.listeners.AsyncChatListener;
import org.bukkit.Bukkit;
import org.bukkit.plugin.java.JavaPlugin;

public class ThemisPlugin extends JavaPlugin {

    @Override
    public void onEnable() {

        Bukkit.getPluginManager().registerEvents(new AsyncChatListener(getComponentLogger()), this);

    }
}
