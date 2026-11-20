import sys
import json
import os
import shutil
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
from DatasetLoader import DatasetLoader
from DirectoryManager import DirectoryManager


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
    if not code:
        return code
    if not is_wrapped_in_class(code):
        code = wrap_in_class(code)
    if not has_required_imports(code):
        code = prepend_imports(code)
    return code


def load_fragment_text(jh: JSONHandler, fragment_id: str) -> str:
    frag = jh.get_by_id(fragment_id)
    if not frag:
        raise ValueError(f"Fragment '{fragment_id}' not found in book data")
    return frag['text']


def load_optimal_context(jh: JSONHandler, knapsack_path: str) -> str:
    with open(knapsack_path, 'r', encoding='utf-8') as f:
        k_data = json.load(f)

    selected_ids = [item['id'] for item in k_data.get('selected_fragments', [])]

    mega_text = ""
    for f_id in selected_ids:
        frag = jh.get_by_id(f_id)
        if frag:
            mega_text += frag['text'] + "\n\n"
    return mega_text.strip()


def find_best_single_paragraph(model_id: str, dataset_name: str, book: str, strategy: str, dm: DirectoryManager) -> dict:
    pred_filename = f"predictions_{book}_{strategy}_{dataset_name}_{model_id}.json"
    pred_path = dm.get_output_path(pred_filename)

    if not Path(pred_path).exists():
        print(f"[Auto-detect] No predictions file found: {pred_filename}")
        return None

    with open(pred_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    fragment_results = {}
    for pred in predictions:
        frag_id = pred.get("fragment_id")
        if not frag_id:
            continue
        if frag_id not in fragment_results:
            fragment_results[frag_id] = {"total": 0, "passed": 0}
        fragment_results[frag_id]["total"] += 1
        fragment_results[frag_id]["passed"] += pred.get("test_status", 0)

    if not fragment_results:
        print(f"[Auto-detect] No fragment results found in {pred_filename}")
        return None

    best_frag_id = max(fragment_results.keys(), key=lambda fid: fragment_results[fid]["passed"] / fragment_results[fid]["total"] if fragment_results[fid]["total"] > 0 else 0)
    best_stats = fragment_results[best_frag_id]
    best_rate = best_stats["passed"] / best_stats["total"] if best_stats["total"] > 0 else 0

    print(f"[Auto-detect] Best single paragraph: fragment_id={best_frag_id}, pass_rate={best_rate:.2f} ({best_stats['passed']}/{best_stats['total']})")

    return {
        "strategy": strategy,
        "fragment_id": best_frag_id
    }


def find_best_optimal_prompt(model_id: str, dataset_name: str, book: str, strategy: str, token_limits: list, dm: DirectoryManager) -> dict:
    best_limit = None
    best_value = -1

    for limit in token_limits:
        knapsack_filename = f"knapsack_{book}_{strategy}_{limit}_{dataset_name}.json"
        knapsack_path = dm.get_output_path(knapsack_filename)

        if not Path(knapsack_path).exists():
            continue

        with open(knapsack_path, 'r', encoding='utf-8') as f:
            k_data = json.load(f)

        total_value = k_data.get("total_value", 0)
        if total_value > best_value:
            best_value = total_value
            best_limit = limit

    if best_limit is None:
        print(f"[Auto-detect] No knapsack files found for {book}_{strategy}")
        return None

    print(f"[Auto-detect] Best optimal prompt: token_limit={best_limit}, total_value={best_value}")

    return {
        "strategy": strategy,
        "token_limit": best_limit
    }


def update_settings_with_best_config(model_id: str, dataset_name: str, book: str, sp_result: dict, opt_result: dict):
    settings_path = Path("configs/settings.json")
    if not settings_path.exists():
        return

    with open(settings_path, 'r', encoding='utf-8') as f:
        settings = json.load(f)

    pc_config = settings.get("prompt_comparison", {})
    configurations = pc_config.get("configurations", {})

    config_key = f"{model_id}_{dataset_name}_{book}"

    if config_key not in configurations:
        configurations[config_key] = {
            "model_id": model_id,
            "dataset": dataset_name,
            "book": book,
            "best_single_paragraph": {},
            "best_optimal_prompt": {}
        }

    if sp_result:
        configurations[config_key]["best_single_paragraph"] = sp_result
        print(f"[Settings] Updated best_single_paragraph for {config_key}")

    if opt_result:
        configurations[config_key]["best_optimal_prompt"] = opt_result
        print(f"[Settings] Updated best_optimal_prompt for {config_key}")

    pc_config["configurations"] = configurations
    settings["prompt_comparison"] = pc_config

    with open(settings_path, 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)

    print(f"[Settings] Saved updated settings to configs/settings.json")


def main():
    cfg = ConfigManager()

    env_path = cfg.dm.get_config_path(".env")
    load_dotenv(env_path)

    dataset_name, dataset_path = cfg.get_dataset_info()
    models = cfg.get_models(only_enabled=True)

    pc_config = cfg.get_prompt_comparison_config()
    validation_split = pc_config.get("validation_split", "val")

    simple_template = pc_config.get("simple_baseline", {}).get("prompt_template", "Implement this:\n{task_prompt}")

    enabled_books = cfg.get_books(only_enabled=True)
    if not enabled_books:
        print("[!] No enabled books found in settings.json")
        return

    default_book = enabled_books[0]["id"]
    token_limits = cfg.knapsack_limits

    if not models:
        print("[!] No active models found in settings.json")
        return

    loader = DatasetLoader(cfg.dm, cfg)
    val_tasks = loader.load_dataset(dataset_name, split=validation_split)
    print(f"\n[Dataset] Loaded {validation_split} split: {len(val_tasks)} tasks (dataset: {dataset_name})", flush=True)

    for model_cfg in models:
        model_id = model_cfg["id"]
        provider = model_cfg["provider"]
        model_name = model_cfg["model_name"]

        for book_cfg in enabled_books:
            book = book_cfg["id"]
            strategy = "paragraph"

            config_key = f"{model_id}_{dataset_name}_{book}"
            configurations = pc_config.get("configurations", {})
            model_config = configurations.get(config_key, {})

            sp_cfg = model_config.get("best_single_paragraph", {})
            opt_cfg = model_config.get("best_optimal_prompt", {})

            sp_fragment_id = sp_cfg.get("fragment_id") if sp_cfg else None
            opt_token_limit = opt_cfg.get("token_limit") if opt_cfg else None

            if not sp_fragment_id:
                print(f"[Auto-detect] No best_single_paragraph for {config_key}, searching...")
                sp_result = find_best_single_paragraph(model_id, dataset_name, book, strategy, cfg.dm)
                if sp_result:
                    sp_fragment_id = sp_result.get("fragment_id")
                    update_settings_with_best_config(model_id, dataset_name, book, sp_result, None)
                    sp_cfg = sp_result

            if not opt_token_limit:
                print(f"[Auto-detect] No best_optimal_prompt for {config_key}, searching...")
                opt_result = find_best_optimal_prompt(model_id, dataset_name, book, strategy, token_limits, cfg.dm)
                if opt_result:
                    opt_token_limit = opt_result.get("token_limit")
                    update_settings_with_best_config(model_id, dataset_name, book, None, opt_result)
                    opt_cfg = opt_result

            if not sp_fragment_id:
                print(f"[!] Cannot determine best_single_paragraph for {config_key}, skipping.")
                continue
            if not opt_token_limit:
                print(f"[!] Cannot determine best_optimal_prompt for {config_key}, skipping.")
                continue

            sp_book_json_name = f"{book}_{strategy}.json"
            sp_book_json_path = cfg.dm.get_output_path(sp_book_json_name)

            opt_book_json_name = f"{book}_{strategy}.json"
            opt_book_json_path = cfg.dm.get_output_path(opt_book_json_name)

            opt_knapsack_filename = f"knapsack_{book}_{strategy}_{opt_token_limit}_{dataset_name}.json"
            opt_knapsack_path = cfg.dm.get_output_path(opt_knapsack_filename)

            if not Path(sp_book_json_path).exists():
                print(f"[!] Book file {sp_book_json_name} is missing, cannot load single-paragraph prompt.")
                continue
            if not Path(opt_book_json_path).exists():
                print(f"[!] Book file {opt_book_json_name} is missing, cannot load optimal prompt.")
                continue
            if not Path(opt_knapsack_path).exists():
                print(f"[!] Knapsack file {opt_knapsack_filename} is missing, cannot load optimal prompt.")
                continue

            jh = JSONHandler()
            jh.load(opt_book_json_path)
            sg = SafeGuard(jh)

            single_para_text = load_fragment_text(jh, sp_fragment_id) if sp_fragment_id else ""
            optimal_text = load_optimal_context(jh, opt_knapsack_path)

            print(f"\n[Configuration]")
            print(f"  Model: {model_id}", flush=True)
            print(f"  Dataset: {dataset_name}", flush=True)
            print(f"  Book: {book}", flush=True)
            print(f"  Validation split: {validation_split}", flush=True)
            print(f"  Simple baseline template: {simple_template[:60]}...", flush=True)
            print(f"  Single-paragraph: book={book}, fragment_id={sp_fragment_id}", flush=True)
            print(f"  Optimal: book={book}, token_limit={opt_token_limit}", flush=True)

            print(f"\n{'=' * 60}", flush=True)
            print(f"Prompt Comparison: Model={model_id}, Book={book}", flush=True)
            print(f"{'=' * 60}", flush=True)

            predictor = Predictor(jh, sg, dataset_path, provider=provider, model_name=model_name)
            predictor.dataset_sample = val_tasks
            evaluator = Evaluator(jh, cfg.dm)

            results = {
                "simple_baseline": [],
                "best_single_paragraph": [],
                "best_optimal_prompt": []
            }

            total = len(val_tasks)
            for idx, task in enumerate(val_tasks, 1):
                t_id = task['task_id']
                test_code = task.get('test', '')
                print(f"[{idx}/{total}] Evaluating: {t_id}", flush=True)

                sp_prompt = predictor.build_prompt(single_para_text, task['prompt'])
                sp_check = sg.is_fragment_safe(sp_prompt)
                if sp_check["is_safe"]:
                    sp_raw, _ = predictor.get_llm_prediction(sp_prompt)
                    sp_code = predictor.clean_llm_output(sp_raw) if sp_raw else ""
                else:
                    sp_code = ""

                if sp_code:
                    sp_code = prepare_code_for_compilation(sp_code)
                sp_comp, sp_test = evaluator.run_java_test(sp_code, test_code)

                results["best_single_paragraph"].append({
                    "task_id": t_id,
                    "compilation": sp_comp,
                    "test_pass": sp_test
                })

                opt_prompt = predictor.build_prompt(optimal_text, task['prompt'])
                opt_check = sg.is_fragment_safe(opt_prompt)
                if opt_check["is_safe"]:
                    opt_raw, _ = predictor.get_llm_prediction(opt_prompt)
                    opt_code = predictor.clean_llm_output(opt_raw) if opt_raw else ""
                else:
                    opt_code = ""

                if opt_code:
                    opt_code = prepare_code_for_compilation(opt_code)
                opt_comp, opt_test = evaluator.run_java_test(opt_code, test_code)

                results["best_optimal_prompt"].append({
                    "task_id": t_id,
                    "compilation": opt_comp,
                    "test_pass": opt_test
                })

                simple_prompt = simple_template.format(task_prompt=task['prompt'])
                simple_check = sg.is_fragment_safe(simple_prompt)
                if simple_check["is_safe"]:
                    simple_raw, _ = predictor.get_llm_prediction(simple_prompt)
                    simple_code = predictor.clean_llm_output(simple_raw) if simple_raw else ""
                else:
                    simple_code = ""

                if simple_code:
                    simple_code = prepare_code_for_compilation(simple_code)
                simple_comp, simple_test = evaluator.run_java_test(simple_code, test_code)

                results["simple_baseline"].append({
                    "task_id": t_id,
                    "compilation": simple_comp,
                    "test_pass": simple_test
                })

                print(f"  [{idx}/{total}] {t_id} done", flush=True)

            def aggregate(res_list):
                total_t = len(res_list)
                comp = sum(x['compilation'] for x in res_list)
                tests = sum(x['test_pass'] for x in res_list)
                return {"total_tasks": total_t, "successful_compilations": comp, "passed_tests": tests}

            final_report = {
                "model_id": model_id,
                "dataset": dataset_name,
                "book": book,
                "split": validation_split,
                "simple_baseline_template": simple_template,
                "single_paragraph": {
                    "book": book,
                    "strategy": strategy,
                    "fragment_id": sp_fragment_id
                },
                "optimal_prompt": {
                    "book": book,
                    "strategy": strategy,
                    "token_limit": opt_token_limit,
                    "knapsack_file": opt_knapsack_filename
                },
                "summary": {
                    "simple_baseline": aggregate(results["simple_baseline"]),
                    "best_single_paragraph": aggregate(results["best_single_paragraph"]),
                    "best_optimal_prompt": aggregate(results["best_optimal_prompt"])
                },
                "details": results
            }

            output_filename = f"prompt_comparison_{validation_split}_{dataset_name}_{model_id}_{book}.json"
            output_path = cfg.dm.get_output_path(output_filename)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_report, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())

            print(f"\n{'=' * 60}", flush=True)
            print(f"Results for {model_id}/{book} ({validation_split} split):", flush=True)
            print(f"  Simple Baseline:      {final_report['summary']['simple_baseline']['successful_compilations']}/{final_report['summary']['simple_baseline']['total_tasks']} comp, {final_report['summary']['simple_baseline']['passed_tests']}/{final_report['summary']['simple_baseline']['total_tasks']} tests", flush=True)
            print(f"  Best Single-Paragraph: {final_report['summary']['best_single_paragraph']['successful_compilations']}/{final_report['summary']['best_single_paragraph']['total_tasks']} comp, {final_report['summary']['best_single_paragraph']['passed_tests']}/{final_report['summary']['best_single_paragraph']['total_tasks']} tests", flush=True)
            print(f"  Best Optimal Prompt:  {final_report['summary']['best_optimal_prompt']['successful_compilations']}/{final_report['summary']['best_optimal_prompt']['total_tasks']} comp, {final_report['summary']['best_optimal_prompt']['passed_tests']}/{final_report['summary']['best_optimal_prompt']['total_tasks']} tests", flush=True)
            print(f"  Report saved to: {output_filename}", flush=True)

            if evaluator.temp_dir.exists():
                shutil.rmtree(evaluator.temp_dir)
                print(f"--- Temp directory {evaluator.temp_dir} deleted. ---")


if __name__ == "__main__":
    main()
