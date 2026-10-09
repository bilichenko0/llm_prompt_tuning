# import sys
# from pathlib import Path

# core_path = str(Path(__file__).parent.parent / "core")
# if core_path not in sys.path:
#     sys.path.append(core_path)

# from DirectoryManager import DirectoryManager
# from JSONHandler import JSONHandler
# from Evaluator import Evaluator

# def main():
#     dm = DirectoryManager()
#     output_path = dm.get_output_path("effective_java_data.json")
#     dataset_path = dm.get_dataset_path("humaneval.jsonl")
#     predictions_path = dm.get_output_path("predictions.json")

#     jh = JSONHandler()
#     print(f"Loading from: {output_path}")  # Add this line to debug
#     jh.load(output_path)

#     ev = Evaluator(jh, dm)
#     ev.evaluate_all(predictions_path, dataset_path)

#     jh.save(output_path)
#     print("Evaluation finished. Metrics updated.")

# if __name__ == "__main__":
#     main()


import sys
from pathlib import Path

core_path = str(Path(__file__).parent.parent / "core")
if core_path not in sys.path:
    sys.path.append(core_path)

from ConfigManager import ConfigManager
from JSONHandler import JSONHandler
from Evaluator import Evaluator

def main():
    cfg = ConfigManager()
    jh = JSONHandler()
    ev = Evaluator(jh, cfg.dm)

    dataset_name, dataset_path = cfg.get_dataset_info()
    books = cfg.get_books(only_enabled=True)
    strategies = cfg.get_fragmentation_strategies()
    models = cfg.get_models(only_enabled=True)

    for book in books:
        for strategy in strategies:
            book_json_name = f"{book['id']}_{strategy}.json"
            book_json_path = cfg.dm.get_output_path(book_json_name)

            if not Path(book_json_path).exists():
                print(f"[!] File {book_json_name} is missing, skipping.")
                continue

            for model_cfg in models:
                model_id = model_cfg["id"]
                pred_filename = f"predictions_{book['id']}_{strategy}_{dataset_name}_{model_id}.json"
                pred_path = cfg.dm.get_output_path(pred_filename)

                if not Path(pred_path).exists():
                    print(f"[!] File {pred_filename} not found, skipping.")
                    continue

                print(f"\nEvaluating: {book_json_name} on predictions from {model_id} ({dataset_name})...")
                jh.load(book_json_path)
                ev.evaluate_all(pred_path, dataset_path)
                jh.save(book_json_path)
                print(f"Metrics for {book_json_name} successfully saved.")

if __name__ == "__main__":
    main()