import json
import os

CONFIG_FILE = "config.json"

DEFAULT_CONFIG = {
    "openai_api_key": "",
    "google_api_key": "",
    "custom_model_paths": [],
    "theme_preference": "dark",
    "use_gpu": True,
    "last_selected_llm_provider": "local", # local, ollama, openai, google
    "last_selected_llm_model": "",
    "last_selected_embed_provider": "local",
    "last_selected_embed_model": ""
}

class ConfigManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance.config = cls._instance.load_config()
        return cls._instance

    def load_config(self):
        if not os.path.exists(CONFIG_FILE):
            self.save_config(DEFAULT_CONFIG)
            return DEFAULT_CONFIG.copy()
        
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
                # Merge with default to ensure all keys exist
                for key, value in DEFAULT_CONFIG.items():
                    if key not in config:
                        config[key] = value
                return config
        except Exception as e:
            print(f"Error loading config: {e}")
            return DEFAULT_CONFIG.copy()

    def save_config(self, config=None):
        if config is not None:
            self.config = config
        
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value
        self.save_config()
