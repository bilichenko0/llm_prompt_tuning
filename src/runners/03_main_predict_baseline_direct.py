#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script makes direct baseline predictions without any literature fragments.
It uses the get_llm_prediction method directly for each task in the dataset.
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
    predictions_path = dm.get_output_path("predictions_baseline_direct.json")
    
    # Initialize JSON handler
    jh = JSONHandler()
    jh.load(output_path)
    sg = SafeGuard(jh)
    
    # Initialize predictor
    predictor = Predictor(jh, sg, dataset_path, provider=CURRENT_PROVIDER, model_name=CURRENT_MODEL)
    
    # Load dataset sample
    dataset_sample = predictor._load_dataset_sample(limit=40)  # Adjust limit as needed
    
    # Create predictions list
    predictions_data = []
    
    # Process each task in the dataset
    for i, task in enumerate(dataset_sample):
        # Build basic prompt without any fragment advice
        prompt_text = f"""You are an expert Java developer.\nComplete the following Java code:\n{task['prompt']}"""
        
        # Get prediction from LLM
        predicted_code = predictor.get_llm_prediction(prompt_text)
        if predicted_code:
            clean_code = predictor.clean_llm_output(predicted_code)
            
            # Create prediction item
            prediction_item = {
                "fragment_id": "baseline",  # Mark as baseline prediction
                "task_id": task["task_id"],
                "predicted_code": clean_code,
                "status": "pending_evaluation" 
            }
            predictions_data.append(prediction_item)
            
            # Save progress periodically
            if (i + 1) % 10 == 0 or i == len(dataset_sample) - 1:
                jh.save(predictions_path, predictions_data)
                print(f"Processed {i + 1}/{len(dataset_sample)} tasks")
    
    # Final save
    jh.save(predictions_path, predictions_data)
    print(f"Predictions finished using {CURRENT_PROVIDER}, {CURRENT_MODEL}. Saved to {predictions_path}")


if __name__ == "__main__":
    main()