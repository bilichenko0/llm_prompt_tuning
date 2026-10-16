#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script makes predictions without any literature fragments and saves them in a separate file.
It is equivalent to 03_main_predict.py but configured for baseline predictions.
"""
import sys
from pathlib import Path
import json

# Add core directory to path
core_path = str(Path(__file__).parent.parent / "core")
if core_path not in sys.path:
    sys.path.append(core_path)

# Import required modules
from DirectoryManager import DirectoryManager
from JSONHandler import JSONHandler
from SafeGuard import SafeGuard
from Predictor import Predictor

from dotenv import load_dotenv


def main():
    # Initialize directory manager
    dm = DirectoryManager()
    
    # Load environment variables
    env_path = dm.get_config_path(".env")
    load_dotenv(env_path)
    
    # Load settings
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
    
    # Define paths for data and output
    output_path = dm.get_output_path("effective_java_data.json")
    dataset_path = dm.get_dataset_path("humaneval.jsonl")
    # Use a separate file for baseline predictions
    predictions_path = dm.get_output_path("predictions_baseline.json")
    
    # Initialize JSON handler
    jh = JSONHandler()
    jh.load(output_path)
    sg = SafeGuard(jh)
    
    # Initialize predictor with baseline configuration
    predictor = Predictor(jh, sg, dataset_path, provider=CURRENT_PROVIDER, model_name=CURRENT_MODEL)
    
    # Run predictions with no fragments (empty fragment text will be used)
    predictor.run_predictions(predictions_path, fragments_limit=410)
    
    print(f"Predictions finished using {CURRENT_PROVIDER}, {CURRENT_MODEL}. Saved to {predictions_path}")


if __name__ == "__main__":
    main()