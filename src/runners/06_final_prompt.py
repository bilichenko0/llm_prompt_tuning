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
from Evaluator import Evaluator
from FinalPromptTester import FinalPromptTester
from DatasetLoader import DatasetLoader


def has_required_imports(code: str) -> bool:
    lines = code.strip().split('\n')
    has_util = any('import java.util.*;' in l.strip() for l in lines[:10])
    has_lang = any('import java.lang.*;' in l.strip() for l in lines[:10])
    return has_util and has_lang


def prepend_imports(code: str) -> str:
    lines = code.split('\n')
    lines_to_keep = [l for l in lines if l.strip() not in ('import java.util.*;', 'import java.lang.*;')]
    
    class_idx = next((i for i, l in enumerate(lines_to_keep) if l.strip().startswith('class ')), -1)
    
    if class_idx >= 0:
        result_lines = lines_to_keep[:class_idx] + ['import java.util.*;', 'import java.lang.*;', ''] + lines_to_keep[class_idx:]
    else:
        result_lines = ['import java.util.*;', 'import java.lang.*;', ''] + lines_to_keep
        
    return '\n'.join(result_lines).strip()


def is_wrapped_in_class(code: str) -> bool:
    return any(l.strip().startswith(('class ', 'public class ')) for l in code.strip().split('\n'))


def wrap_in_class(code: str) -> str:
    lines = code.split('\n')
    import_lines, code_lines = [], []
    for line in lines:
        if line.strip().startswith('import '):
            import_lines.append(line)
        else:
            code_lines.append(line)
            
    indented = ['    ' + l if l.strip() else l for l in code_lines]
    return '\n'.join(import_lines + [''] + ['class Solution {'] + indented + ['}'])


def prepare_code_for_compilation(code: str) -> str:
    if not is_wrapped_in_class(code):
        code = wrap_in_class(code)
    if not has_required_imports(code):
        code = prepend_imports(code)
    return code


def main():
    cfg = ConfigManager()
    
    env_path = cfg.dm.get_config_path(".env")
    load_dotenv(env_path)

    dataset_name, dataset_path = cfg.get_dataset_info()
    models = cfg.get_models(only_enabled=True)

    loader = DatasetLoader(cfg.dm, cfg)
    test_tasks = loader.load_dataset(dataset_name, split="train")
    print(f"\n[Dataset] Loaded train split: {len(test_tasks)} tasks (dataset: {dataset_name})", flush=True)

    books = cfg.get_books(only_enabled=True)
    strategies = cfg.get_fragmentation_strategies()
    token_limits = cfg.knapsack_limits
    sample_limit = cfg.get_sample_limit()

    if not models:
        print("[!] No active models found in settings.json")
        return

    for book in books:
        for strategy in strategies:
            book_json_name = f"{book['id']}_{strategy}.json"
            book_json_path = cfg.dm.get_output_path(book_json_name)

            if not Path(book_json_path).exists():
                print(f"[!] Book file {book_json_name} is missing, skipping.")
                continue

            jh = JSONHandler()
            jh.load(book_json_path)
            sg = SafeGuard(jh)

            for limit in token_limits:
                knapsack_results_filename = f"knapsack_{book['id']}_{strategy}_{limit}_{dataset_name}.json"
                knapsack_results_path = cfg.dm.get_output_path(knapsack_results_filename)

                if not Path(knapsack_results_path).exists():
                    print(f"[!] Knapsack file {knapsack_results_filename} has not been generated yet, skipping.")
                    continue

                for model_cfg in models:
                    provider = model_cfg["provider"]
                    model_name = model_cfg["model_name"]
                    model_id = model_cfg["id"]

                    final_report_filename = f"final_ab_test_report_{book['id']}_{strategy}_{limit}_{dataset_name}_{model_id}.json"
                    final_report_path = cfg.dm.get_output_path(final_report_filename)

                    print(f"\n==================================================", flush=True)
                    print(f"A/B Test: {book['id']} ({strategy}) | Limit: {limit} | Model: {model_id}", flush=True)
                    print(f"==================================================")

                    predictor = Predictor(jh, sg, dataset_path, provider=provider, model_name=model_name, sample_limit=sample_limit)
                    predictor.dataset_sample = test_tasks
                    evaluator = Evaluator(jh, cfg.dm)
                    
                    tester = FinalPromptTester(predictor, evaluator, jh, cfg.dm, code_preparer=prepare_code_for_compilation)
                    
                    report = tester.run_ab_test(knapsack_results_path, final_report_path)
                    
                    print(f"\nResults for {model_id} (context limit {limit}):")
                    print(f"  Baseline: {report['summary_baseline']['passed_tests']}/{report['summary_baseline']['total_tasks']} tests")
                    print(f"  Mega-Prompt: {report['summary_mega_prompt']['passed_tests']}/{report['summary_mega_prompt']['total_tasks']} tests")
                    print(f"Report saved to: {final_report_filename}", flush=True)


if __name__ == "__main__":
    main()