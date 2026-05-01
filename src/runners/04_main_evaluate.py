import sys
from pathlib import Path

core_path = str(Path(__file__).parent.parent / "core")
if core_path not in sys.path:
    sys.path.append(core_path)

from DirectoryManager import DirectoryManager
from JSONHandler import JSONHandler
from Evaluator import Evaluator

def main():
    dm = DirectoryManager()
    output_path = dm.get_output_path("effective_java_data.json")
    dataset_path = dm.get_dataset_path("humaneval.jsonl")
    predictions_path = dm.get_output_path("predictions.json")

    jh = JSONHandler()
    jh.load(output_path)

    ev = Evaluator(jh, dm)
    ev.evaluate_all(predictions_path, dataset_path)

    jh.save(output_path)
    print("Evaluation finished. Metrics updated.")

if __name__ == "__main__":
    main()