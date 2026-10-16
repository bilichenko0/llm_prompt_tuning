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
    sample_limit = cfg.get_sample_limit()
    
    if not models:
        print("[!] No active models found in settings.json")
        return

    for model_cfg in models:
        provider = model_cfg["provider"]
        model_name = model_cfg["model_name"]
        model_id = model_cfg["id"]

        predictions_filename = f"predictions_baseline_direct_{dataset_name}_{model_id}.json"
        predictions_path = cfg.dm.get_output_path(predictions_filename)

        print(f"\n[Baseline Direct] Running: {model_id} ({model_name}) | Dataset: {dataset_name} | Tasks: {sample_limit}")

        sg = SafeGuard(jh)
        predictor = Predictor(jh, sg, dataset_path, provider=provider, model_name=model_name)
        dataset_sample = predictor._load_dataset_sample(limit=sample_limit)

        predictions_data = []

        for i, task in enumerate(dataset_sample):
            prompt_text = f"You are an expert Java developer.\nComplete the following Java code:\n{task['prompt']}"
            predicted_code, meta = predictor.get_llm_prediction(prompt_text)
            
            if predicted_code:
                clean_code = predictor.clean_llm_output(predicted_code)
                prediction_item = {
                    "fragment_id": "baseline",
                    "task_id": task["task_id"],
                    "predicted_code": clean_code,
                    "status": "pending_evaluation",
                    "llm_metadata": meta
                }
                predictions_data.append(prediction_item)

            if (i + 1) % 5 == 0 or i == len(dataset_sample) - 1:
                jh.save(predictions_path, predictions_data)
                print(f"Processed {i + 1}/{len(dataset_sample)} tasks")

        jh.save(predictions_path, predictions_data)
        print(f"Saved to: {predictions_path}")

if __name__ == "__main__":
    main()