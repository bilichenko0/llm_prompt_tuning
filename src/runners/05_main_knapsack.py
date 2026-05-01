import sys
from pathlib import Path

core_path = str(Path(__file__).parent.parent / "core")
if core_path not in sys.path:
    sys.path.append(core_path)

import os
import json
from DirectoryManager import DirectoryManager
from JSONHandler import JSONHandler
from KnapsackSolver import KnapsackSolver

def main():
    dm = DirectoryManager()
    
    # config_path = dm.project_root / "knapsack_config.json"

    config_path = dm.get_config_path("knapsack_config.json")
    output_path = dm.get_output_path("effective_java_data.json")
    knapsack_output_path = dm.get_output_path("knapsack_results.json")

    default_config = {
        "token_limit": 1200,
        "target_metric": "test_pass_rate",  #compilation, test_pass_rate, code_bleu
        "model_base": "cl100k_base"         #cl100k_base, o200k_base
    }


    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = default_config
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    TOKEN_LIMIT = config.get("token_limit", 1200)
    TARGET_METRIC = config.get("target_metric", "test_pass_rate")
    MODEL_BASE = config.get("model_base", "cl100k_base")

    jh = JSONHandler()
    jh.load(output_path)

    solver = KnapsackSolver(jh, TOKEN_LIMIT, metric_name=TARGET_METRIC)
    best_fragment_ids = solver.solve()

    total_value = 0
    total_tokens = 0
    results_data = []

    for f_id in best_fragment_ids:
        item = jh.get_by_id(f_id)
        val = item["metadata"]["metrics"].get(TARGET_METRIC) or 0.0
        tok = item["metadata"]["tokens"].get(MODEL_BASE) or 0
        
        total_value += val
        total_tokens += tok
        
        results_data.append({
            "id": f_id,
            "weight_metric": val,
            "tokens": tok
        })

    summary = {
        "status": "success",
        "config_used": config,
        "total_selected": len(best_fragment_ids),
        "total_value": total_value,
        "total_tokens": total_tokens,
        "selected_fragments": results_data
    }

    with open(knapsack_output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()