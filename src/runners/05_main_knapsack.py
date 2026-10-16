import sys
from pathlib import Path
import json

core_path = str(Path(__file__).parent.parent / "core")
if core_path not in sys.path:
    sys.path.append(core_path)

from ConfigManager import ConfigManager
from JSONHandler import JSONHandler
from KnapsackSolver import KnapsackSolver

def main():
    cfg = ConfigManager()
    jh = JSONHandler()

    dataset_name, _ = cfg.get_dataset_info()
    target_metric = cfg.knapsack_target_metric
    token_limits = cfg.knapsack_limits
    model_base = cfg.data.get("knapsack", {}).get("model_base", "cl100k_base")

    books = cfg.get_books(only_enabled=True)
    strategies = cfg.get_fragmentation_strategies()

    print(f"Starting Knapsack | Metric: {target_metric} | Limit: {token_limits}")

    for book in books:
        for strategy in strategies:
            book_json_name = f"{book['id']}_{strategy}.json"
            book_json_path = cfg.dm.get_output_path(book_json_name)

            if not Path(book_json_path).exists():
                print(f"{book_json_name} does not exist.")
                continue

            jh.load(book_json_path)

            for limit in token_limits:
                
                solver = KnapsackSolver(jh, limit, metric_name=target_metric, model_base=model_base)
                best_fragment_ids = solver.solve()

                total_value = 0.0
                total_tokens = 0
                results_data = []

                for f_id in best_fragment_ids:
                    item = jh.get_by_id(f_id)
                    if not item:
                        continue
                    
                    val = item.get("metadata", {}).get("metrics", {}).get(target_metric) or 0.0
                    tok = item.get("metadata", {}).get("tokens", {}).get(model_base) or 0
                    
                    total_value += val
                    total_tokens += tok
                    
                    results_data.append({
                        "id": f_id,
                        "weight_metric": val,
                        "tokens": tok
                    })

                summary = {
                    "status": "success",
                    "book": book["id"],
                    "strategy": strategy,
                    "target_metric": target_metric,
                    "token_limit": limit,
                    "model_base": model_base,
                    "total_selected": len(best_fragment_ids),
                    "total_value": total_value,
                    "total_tokens": total_tokens,
                    "selected_fragments": results_data
                }

                out_filename = f"knapsack_{book['id']}_{strategy}_{limit}_{dataset_name}.json"
                out_path = cfg.dm.get_output_path(out_filename)

                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)

                print(f"Result is saved do {out_filename} ({len(best_fragment_ids)} fragments)")

if __name__ == "__main__":
    main()