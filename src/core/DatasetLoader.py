import json
from pathlib import Path
from DirectoryManager import DirectoryManager
from ConfigManager import ConfigManager


class DatasetLoader:
    def __init__(self, dm: DirectoryManager = None, cfg: ConfigManager = None):
        self.dm = dm or DirectoryManager()
        self.cfg = cfg or ConfigManager()
        self._split_cache: dict[str, set] = {}

    def _get_split_path(self, split: str) -> Path:
        statistics_dir = self.dm.project_root / "statistics"
        return statistics_dir / f"split_{split}.json"

    def load_split_task_ids(self, split: str) -> set[str]:
        if split in self._split_cache:
            return self._split_cache[split]

        split_path = self._get_split_path(split)
        if not split_path.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_path}. "
                f"Run stratified_split.py to generate split files."
            )

        with open(split_path, "r", encoding="utf-8") as f:
            task_ids = json.load(f)

        self._split_cache[split] = set(task_ids)
        return self._split_cache[split]

    def _resolve_dataset_path(self, dataset_name: str, use_mini: bool = False) -> str:
        items = self.cfg.data.get("datasets", {}).get("items", {})
        info = items.get(dataset_name, {})
        if not info:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        filename = info.get("mini_file") if use_mini else info.get("full_file")
        if not filename:
            raise ValueError(f"No file configured for dataset: {dataset_name}")

        return self.dm.get_dataset_path(filename)

    @staticmethod
    def _load_records(dataset_path: str) -> list[dict]:
        records = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def load_dataset(self, dataset_name: str, split: str = "all", use_mini: bool = False) -> list[dict]:
        dataset_path = self._resolve_dataset_path(dataset_name, use_mini)
        records = self._load_records(dataset_path)

        if split is None or split == "all":
            return records

        split_task_ids = self.load_split_task_ids(split)
        return [r for r in records if r.get("task_id") in split_task_ids]
