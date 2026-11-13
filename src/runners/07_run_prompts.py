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


def main():
    cfg = ConfigManager()

    env_path = cfg.dm.get_config_path(".env")
    load_dotenv(env_path)

    dataset_name, dataset_path = cfg.get_dataset_info()
    models = cfg.get_models(only_enabled=True)

    pc_config = cfg.get_prompt_comparison_config()
    validation_split = pc_config.get("validation_split", "val")
    target_model_id = pc_config.get("model_id", models[0]["id"] if models else None)

    loader = DatasetLoader(cfg.dm, cfg)
    val_tasks = loader.load_dataset(dataset_name, split=validation_split)
    print(f"\n[Dataset] Loaded {validation_split} split: {len(val_tasks)} tasks (dataset: {dataset_name})", flush=True)

    simple_template = pc_config.get("simple_baseline", {}).get("prompt_template", "Implement this:\n{task_prompt}")

    sp_cfg = pc_config.get("best_single_paragraph", {})
    sp_book = sp_cfg.get("book", "effective_java")
    sp_strategy = sp_cfg.get("strategy", "paragraph")
    sp_fragment_id = sp_cfg.get("fragment_id")

    opt_cfg = pc_config.get("best_optimal_prompt", {})
    opt_book = opt_cfg.get("book", "effective_java")
    opt_strategy = opt_cfg.get("strategy", "paragraph")
    opt_token_limit = opt_cfg.get("token_limit", 700)

    sp_book_json_name = f"{sp_book}_{sp_strategy}.json"
    sp_book_json_path = cfg.dm.get_output_path(sp_book_json_name)

    opt_book_json_name = f"{opt_book}_{opt_strategy}.json"
    opt_book_json_path = cfg.dm.get_output_path(opt_book_json_name)

    opt_knapsack_filename = f"knapsack_{opt_book}_{opt_strategy}_{opt_token_limit}_{dataset_name}.json"
    opt_knapsack_path = cfg.dm.get_output_path(opt_knapsack_filename)

    if not Path(sp_book_json_path).exists():
        print(f"[!] Book file {sp_book_json_name} is missing, cannot load single-paragraph prompt.")
        return
    if not Path(opt_book_json_path).exists():
        print(f"[!] Book file {opt_book_json_name} is missing, cannot load optimal prompt.")
        return
    if not Path(opt_knapsack_path).exists():
        print(f"[!] Knapsack file {opt_knapsack_filename} is missing, cannot load optimal prompt.")
        return

    jh = JSONHandler()
    jh.load(opt_book_json_path)
    sg = SafeGuard(jh)

    single_para_text = load_fragment_text(jh, sp_fragment_id) if sp_fragment_id else ""
    optimal_text = load_optimal_context(jh, opt_knapsack_path)

    if not models:
        print("[!] No active models found in settings.json")
        return

    print(f"\n[Configuration]")
    print(f"  Validation split: {validation_split}", flush=True)
    print(f"  Target model: {target_model_id}", flush=True)
    print(f"  Simple baseline template: {simple_template[:60]}...", flush=True)
    print(f"  Single-paragraph: book={sp_book}, fragment_id={sp_fragment_id}", flush=True)
    print(f"  Optimal: book={opt_book}, token_limit={opt_token_limit}", flush=True)

    for model_cfg in models:
        model_id = model_cfg["id"]
        provider = model_cfg["provider"]
        model_name = model_cfg["model_name"]

        if target_model_id and model_id != target_model_id:
            continue

        print(f"\n{'=' * 60}", flush=True)
        print(f"Prompt Comparison: Model={model_id} ({model_name})", flush=True)
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
            "split": validation_split,
            "simple_baseline_template": simple_template,
            "single_paragraph": {
                "book": sp_book,
                "strategy": sp_strategy,
                "fragment_id": sp_fragment_id
            },
            "optimal_prompt": {
                "book": opt_book,
                "strategy": opt_strategy,
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

        output_filename = f"prompt_comparison_{validation_split}_{dataset_name}_{model_id}.json"
        output_path = cfg.dm.get_output_path(output_filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        print(f"\n{'=' * 60}", flush=True)
        print(f"Results for {model_id} ({validation_split} split):", flush=True)
        print(f"  Simple Baseline:      {final_report['summary']['simple_baseline']['successful_compilations']}/{final_report['summary']['simple_baseline']['total_tasks']} comp, {final_report['summary']['simple_baseline']['passed_tests']}/{final_report['summary']['simple_baseline']['total_tasks']} tests", flush=True)
        print(f"  Best Single-Paragraph: {final_report['summary']['best_single_paragraph']['successful_compilations']}/{final_report['summary']['best_single_paragraph']['total_tasks']} comp, {final_report['summary']['best_single_paragraph']['passed_tests']}/{final_report['summary']['best_single_paragraph']['total_tasks']} tests", flush=True)
        print(f"  Best Optimal Prompt:  {final_report['summary']['best_optimal_prompt']['successful_compilations']}/{final_report['summary']['best_optimal_prompt']['total_tasks']} comp, {final_report['summary']['best_optimal_prompt']['passed_tests']}/{final_report['summary']['best_optimal_prompt']['total_tasks']} tests", flush=True)
        print(f"  Report saved to: {output_filename}", flush=True)

        if evaluator.temp_dir.exists():
            shutil.rmtree(evaluator.temp_dir)
            print(f"--- Temp directory {evaluator.temp_dir} deleted. ---")


if __name__ == "__main__":
    main()
