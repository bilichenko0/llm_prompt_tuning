import sys
from pathlib import Path
from dotenv import load_dotenv

core_path = str(Path(__file__).parent.parent / "core")
if core_path not in sys.path:
    sys.path.append(core_path)

from ConfigManager import ConfigManager
from JSONHandler import JSONHandler
from SafeGuard import SafeGuard
from Predictor import Predictor

def main():
    cfg = ConfigManager()
    jh = JSONHandler()

    env_path = cfg.dm.get_config_path(".env")
    load_dotenv(env_path)

    dataset_name, dataset_path = cfg.get_dataset_info()
    models = cfg.get_models(only_enabled=True)
    books = cfg.get_books(only_enabled=True)
    strategies = cfg.get_fragmentation_strategies()
    fragments_limit = cfg.get_fragments_limit()

    if not models:
        print("[!] No active models found in settings.json")
        return

    for book in books:
        for strategy in strategies:
            book_json_name = f"{book['id']}_{strategy}.json"
            book_json_path = cfg.dm.get_output_path(book_json_name)

            if not Path(book_json_path).exists():
                print(f"[!] File {book_json_name} is missing. Please run parsing and tokenization first.")
                continue

            jh.load(book_json_path)
            sg = SafeGuard(jh)

            for model_cfg in models:
                provider = model_cfg["provider"]
                model_name = model_cfg["model_name"]
                model_id = model_cfg["id"]

                pred_filename = f"predictions_{book['id']}_{strategy}_{dataset_name}_{model_id}.json"
                predictions_path = cfg.dm.get_output_path(pred_filename)

                print(f"\n[Predict Fragments] Book: {book['id']} ({strategy}) | Model: {model_id} | Fragments limit: {fragments_limit}")

                predictor = Predictor(jh, sg, dataset_path, provider=provider, model_name=model_name)
                predictor.run_predictions(predictions_path, fragments_limit=fragments_limit)

                print(f"Done. Predictions saved to: {pred_filename}")

if __name__ == "__main__":
    main()