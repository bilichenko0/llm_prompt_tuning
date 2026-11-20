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


def main():
    cfg = ConfigManager()

    env_path = cfg.dm.get_config_path(".env")
    load_dotenv(env_path)

    dataset_name, dataset_path = cfg.get_dataset_info()
    models = cfg.get_models(only_enabled=True)

    pc_config = cfg.get_prompt_comparison_config()
    simple_template = pc_config.get("simple_baseline", {}).get("prompt_template", "Implement this:\n{task_prompt}")

    loader = DatasetLoader(cfg.dm, cfg)
    train_tasks = loader.load_dataset(dataset_name, split="train")
    print(f"\n[Dataset] Loaded train split: {len(train_tasks)} tasks (dataset: {dataset_name})", flush=True)

    if not models:
        print("[!] No models found in settings.json")
        return

    jh = JSONHandler()
    sg = SafeGuard(jh)

    print(f"\n[Configuration]")
    print(f"  Prompt template: {simple_template[:60]}...", flush=True)
    print(f"  Models to screen: {len(models)}", flush=True)
    for m in models:
        status = "enabled" if m.get("enabled") else "disabled"
        print(f"    - {m['id']} ({m['model_name']}) [{status}]", flush=True)

    all_results = {}

    for model_cfg in models:
        model_id = model_cfg["id"]
        provider = model_cfg["provider"]
        model_name = model_cfg["model_name"]

        print(f"\n{'=' * 60}", flush=True)
        print(f"Model Screening: {model_id} ({model_name})", flush=True)
        print(f"{'=' * 60}", flush=True)

        predictor = Predictor(jh, sg, dataset_path, provider=provider, model_name=model_name)
        predictor.dataset_sample = train_tasks
        evaluator = Evaluator(jh, cfg.dm)

        results = []
        total = len(train_tasks)

        for idx, task in enumerate(train_tasks, 1):
            t_id = task['task_id']
            test_code = task.get('test', '')
            print(f"[{idx}/{total}] Evaluating: {t_id}", flush=True)

            prompt = simple_template.format(task_prompt=task['prompt'])

            check = sg.is_fragment_safe(prompt)
            if not check["is_safe"]:
                code = ""
            else:
                raw, _ = predictor.get_llm_prediction(prompt)
                code = predictor.clean_llm_output(raw) if raw else ""

            if code:
                code = prepare_code_for_compilation(code)

            comp, test_pass = evaluator.run_java_test(code, test_code)

            results.append({
                "task_id": t_id,
                "compilation": comp,
                "test_pass": test_pass
            })

            status_str = "OK" if test_pass else ("COMPILED" if comp else "FAIL")
            print(f"  [{idx}/{total}] {t_id} done ({status_str})", flush=True)

        total_t = len(results)
        comp_sum = sum(x['compilation'] for x in results)
        test_sum = sum(x['test_pass'] for x in results)

        model_summary = {
            "model_id": model_id,
            "provider": provider,
            "model_name": model_name,
            "dataset": dataset_name,
            "split": "train",
            "total_tasks": total_t,
            "successful_compilations": comp_sum,
            "passed_tests": test_sum,
            "compilation_rate": round(comp_sum / total_t, 4) if total_t else 0.0,
            "test_pass_rate": round(test_sum / total_t, 4) if total_t else 0.0
        }

        output_filename = f"prompt_screen_train_{dataset_name}_{model_id}.json"
        output_path = cfg.dm.get_output_path(output_filename)

        report = {
            "model_id": model_id,
            "provider": provider,
            "model_name": model_name,
            "dataset": dataset_name,
            "split": "train",
            "simple_baseline_template": simple_template,
            "summary": model_summary,
            "details": results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        print(f"\nResults for {model_id}:", flush=True)
        print(f"  Compilation: {comp_sum}/{total_t} ({model_summary['compilation_rate']:.1%})", flush=True)
        print(f"  Test pass:   {test_sum}/{total_t} ({model_summary['test_pass_rate']:.1%})", flush=True)
        print(f"  Report saved to: {output_filename}", flush=True)

        all_results[model_id] = model_summary

        if evaluator.temp_dir.exists():
            shutil.rmtree(evaluator.temp_dir)
            print(f"--- Temp directory {evaluator.temp_dir} deleted. ---")

    summary_filename = f"prompt_screen_train_{dataset_name}_summary.json"
    summary_path = cfg.dm.get_output_path(summary_filename)

    sorted_results = sorted(all_results.values(), key=lambda x: x['test_pass_rate'], reverse=True)

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            "dataset": dataset_name,
            "split": "train",
            "prompt_template": simple_template,
            "model_count": len(all_results),
            "models_ranked": sorted_results
        }, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())

    print(f"\n{'=' * 60}", flush=True)
    print(f"MODEL SCREENING SUMMARY ({dataset_name} train split)", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"{'Model':<25} {'Comp Rate':>10} {'Test Pass':>10}", flush=True)
    print("-" * 55, flush=True)
    for r in sorted_results:
        print(f"{r['model_id']:<25} {r['compilation_rate']:>9.1%} {r['test_pass_rate']:>9.1%}", flush=True)
    print(f"\nSummary saved to: {summary_filename}", flush=True)


if __name__ == "__main__":
    main()
