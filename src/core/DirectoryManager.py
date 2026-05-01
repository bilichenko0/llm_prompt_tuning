import os
from pathlib import Path

class DirectoryManager:
    def __init__(self, base_dir=None):
        if base_dir:
            self.project_root = Path(base_dir)
        else:
            self.project_root = Path(__file__).parent.parent.parent.absolute() 

        self.books_dir = self.project_root / "lib"
        self.output_dir = self.project_root / "processed"
        self.configs_dir = self.project_root / "configs"
        self.dataset_dir = self.project_root / "dataset"
        
        for directory in [self.books_dir, self.output_dir, self.configs_dir, self.dataset_dir]:
            directory.mkdir(exist_ok=True)

    def get_output_path(self, filename: str) -> str:
        return str(self.output_dir / filename)

    def get_book_path(self, filename: str) -> str:
        path = self.books_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Book does not exist: {path}")
        return str(path)

    def get_config_path(self, filename: str) -> Path:
        return self.configs_dir / filename
    
    def get_dataset_path(self, filename: str) -> str:
        path = self.dataset_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Dataset does not exist: {path}")
        return str(path)
