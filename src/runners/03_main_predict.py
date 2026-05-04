import sys
from pathlib import Path
import json

core_path = str(Path(__file__).parent.parent / "core")
if core_path not in sys.path:
    sys.path.append(core_path)

from DirectoryManager import DirectoryManager
from JSONHandler import JSONHandler
from SafeGuard import SafeGuard
from Predictor import Predictor

from dotenv import load_dotenv

def main():
    dm = DirectoryManager()

    env_path = dm.get_config_path(".env")
    load_dotenv(env_path)



    settings_path = dm.get_config_path("settings.json")
    try:
        with open(settings_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)
            config = settings.get("llm_config", {})
            CURRENT_PROVIDER = config.get("provider", "openai")
            CURRENT_MODEL = config.get("model_name", "gpt-3.5-turbo")
    except Exception as e:
        print(f"{e}")
        CURRENT_PROVIDER = "openai"
        CURRENT_MODEL = "gpt-3.5-turbo"

    output_path = dm.get_output_path("effective_java_data.json")
    dataset_path = dm.get_dataset_path("humaneval.jsonl")
    predictions_path = dm.get_output_path("predictions.json")

    jh = JSONHandler()
    jh.load(output_path)
    sg = SafeGuard(jh)

    # premenna pre preklopenie frameworku
    #CURRENT_PROVIDER = "openai" #"openai", "litellm"

    predictor = Predictor(jh, sg, dataset_path, provider=CURRENT_PROVIDER, model_name=CURRENT_MODEL)
    predictor.run_predictions(predictions_path, fragments_limit=10)
    
    print(f"Predictions finished using {CURRENT_PROVIDER}, {CURRENT_MODEL}.")

if __name__ == "__main__":
    main()