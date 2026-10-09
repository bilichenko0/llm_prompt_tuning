import json
from pathlib import Path
from DirectoryManager import DirectoryManager

class ConfigManager:
    def __init__(self, config_filename: str = "settings.json"):
        self.dm = DirectoryManager()
        self.config_path = self.dm.get_config_path(config_filename)
        self.data = self._load()

    def _load(self) -> dict:
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @property
    def is_debug(self) -> bool:
        return self.data.get("project", {}).get("debug_mode", False)

    def get_books(self, only_enabled: bool = True) -> list[dict]:
        books = self.data.get("literature", {}).get("books", [])
        if only_enabled:
            return [b for b in books if b.get("enabled", True)]
        return books

    def get_fragmentation_strategies(self) -> list[str]:
        return self.data.get("literature", {}).get("strategies", ["paragraph"])

    def get_parsing_page_range(self) -> tuple[int, int]:
        lit = self.data.get("literature", {})
        start_page = lit.get("start_page", 0)
        end_page = lit.get("debug_max_pages", 5) if self.is_debug else lit.get("full_max_pages", 1000)
        return start_page, end_page
    
    def get_dataset_info(self) -> tuple[str, Path]:
        ds_cfg = self.data.get("datasets", {})
        active = ds_cfg.get("active_dataset", "humaneval")
        use_mini = ds_cfg.get("use_mini", False)
        
        info = ds_cfg.get("items", {}).get(active, {})
        filename = info.get("mini_file") if use_mini else info.get("full_file")
        return active, self.dm.get_dataset_path(filename)

    def get_models(self, only_enabled: bool = True) -> list[dict]:
        models = self.data.get("models", [])
        if only_enabled:
            return [m for m in models if m.get("enabled", True)]
        return models

    @property
    def knapsack_limits(self) -> list[int]:
        return self.data.get("knapsack", {}).get("token_limits", [1200])

    @property
    def knapsack_target_metric(self) -> str:
        return self.data.get("knapsack", {}).get("target_metric", "test_pass_rate")

    def get_sample_limit(self) -> int:
        ds_cfg = self.data.get("datasets", {})
        limits = ds_cfg.get("sample_limits", {})
        if self.is_debug:
            return limits.get("debug", 3)
        return limits.get("full", 40)


    def get_fragments_limit(self) -> int:
        lit = self.data.get("literature", {})
        return lit.get("debug_fragments_limit", 2) if self.is_debug else lit.get("full_fragments_limit", 410)