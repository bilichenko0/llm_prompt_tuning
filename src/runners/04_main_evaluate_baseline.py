#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script evaluates predictions made with a baseline system prompt (without any literature fragment).
It is equivalent to 04_main_evaluate.py but uses only the basic system prompt.
"""
import sys
from pathlib import Path

# Add core directory to path
core_path = str(Path(__file__).parent.parent / "core")
if core_path not in sys.path:
    sys.path.append(core_path)

# Import required modules
from DirectoryManager import DirectoryManager
from JSONHandler import JSONHandler
from Evaluator import Evaluator


def main():
    # Initialize directory manager
    dm = DirectoryManager()
    
    # Define paths for data and output
    output_path = dm.get_output_path("effective_java_data.json")
    dataset_path = dm.get_dataset_path("humaneval.jsonl")
    predictions_path = dm.get_output_path("predictions.json")
    
    # Initialize JSON handler
    jh = JSONHandler()
    print(f"Loading from: {output_path}")  # Debug line
    jh.load(output_path)
    
    # Initialize evaluator
    ev = Evaluator(jh, dm)
    
    # Evaluate all predictions
    ev.evaluate_all(predictions_path, dataset_path)
    
    # Save updated metrics
    jh.save(output_path)
    print("Evaluation finished. Metrics updated.")


if __name__ == "__main__":
    main()