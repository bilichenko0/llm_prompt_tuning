import subprocess
import os
import json
from pathlib import Path
import shutil
import time
from datetime import datetime
from JSONHandler import JSONHandler
from CodeQualityEvaluator import CodeQualityEvaluator

class Evaluator:
    def __init__(self, literature_jh, directory_manager):
        self.literature_jh = literature_jh
        self.dm = directory_manager
        
        self.temp_dir = Path(self.dm.output_dir) / "temp_java"
        self.temp_dir.mkdir(exist_ok=True)
        
        self.error_log_path = Path(self.dm.output_dir) / "evaluation_errors.log"

        self.quality_evaluator = CodeQualityEvaluator()

    def _load_dataset_as_dict(self, dataset_path):
        dataset = {}
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                dataset[item['task_id']] = item
        return dataset

    def _log_error(self, fragment_id, task_id, error_type, error_details):
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "fragment_id": fragment_id,
            "task_id": task_id,
            "error_type": error_type,
            "error_details": error_details
        }
        
        with open(self.error_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')


    def _save_predictions(self, predictions_path, predictions):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with open(predictions_path, 'w', encoding='utf-8') as f:
                    json.dump(predictions, f, indent=2, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())
                return
            except OSError as e:
                if attempt < max_retries - 1:
                    print('Save failed (attempt ' + str(attempt + 1) + '/' + str(max_retries) + '): ' + str(e) + '. Retrying...')
                    time.sleep(pow(2, attempt))
                else:
                    print('Save failed after ' + str(max_retries) + ' attempts: ' + str(e))
                    raise

    def _prepare_code(self, code: str) -> str:
        if not any(l.strip().startswith(('class ', 'public class ')) for l in code.split('\n')):
            code = "class Solution {\n" + "\n".join("    " + l for l in code.split('\n')) + "\n}"
        if "import java.util.*;" not in code:
            code = "import java.util.*;\nimport java.lang.*;\n" + code
        return code

    def run_java_test(self, solution_code, test_code, fragment_id=None, task_id=None) -> tuple:
        self.temp_dir.mkdir(exist_ok=True)
        solution_code = self._prepare_code(solution_code)

        sol_file = self.temp_dir / "Solution.java"
        main_file = self.temp_dir / "Main.java"

        sol_file = self.temp_dir / "Solution.java"
        main_file = self.temp_dir / "Main.java"

        full_test_code = "import java.util.*;\nimport java.lang.*;\n" + test_code

        with open(sol_file, 'w', encoding='utf-8') as f:
            f.write(solution_code)
        
        with open(main_file, 'w', encoding='utf-8') as f:
            f.write(full_test_code)

        try:
            compile_res = subprocess.run(
                ['javac', 'Solution.java', 'Main.java'],
                cwd=self.temp_dir, capture_output=True, text=True, timeout=15
            )
            if compile_res.returncode != 0:
                error_msg = compile_res.stderr
                print(f"--- Compilation Error ---\n{error_msg}")

                
                if fragment_id and task_id:
                    self._log_error(fragment_id, task_id, "compilation_error", error_msg)
                
                return (0, 0)

            run_res = subprocess.run(
                ['java', 'Main'],
                cwd=self.temp_dir, capture_output=True, text=True, timeout=20
            )
            
            if run_res.returncode != 0:
                # Log test failure
                if fragment_id and task_id:
                    error_msg = run_res.stderr if run_res.stderr else "Test failed with non-zero exit code"
                    self._log_error(fragment_id, task_id, "test_failure", error_msg)
            
            return (1, 1 if run_res.returncode == 0 else 0)

        except subprocess.TimeoutExpired as e:
            error_msg = f"Command '{' '.join(e.cmd)}' timed out after {e.timeout} seconds"
            print(f"--- Timeout Error ---\n{error_msg}")
            
            # Log timeout error
            if fragment_id and task_id:
                self._log_error(fragment_id, task_id, "timeout_error", error_msg)
            
            return (1, 0)  
        except Exception as e:
            error_msg = str(e)
            print(f"--- Execution Error ---\n{error_msg}")
            
            if fragment_id and task_id:
                self._log_error(fragment_id, task_id, "execution_error", error_msg)
            
            return (0, 0)

    def evaluate_all(self, predictions_path: str, dataset_path: str):
        
        with open(predictions_path, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        
        dataset = self._load_dataset_as_dict(dataset_path)

        
        evaluated_count = 0
        skipped_count = 0
        for pred in predictions:
            if pred.get('status') == 'evaluated' and 'compilation_status' in pred and 'test_status' in pred:
                evaluated_count += 1
        
        if evaluated_count > 0:
            print(f"Found {evaluated_count} already evaluated predictions. Resuming from there...")
        
        results_aggregator = {}

        total_predictions = len(predictions)
        for idx, pred in enumerate(predictions, 1):
            f_id = pred['fragment_id']
            t_id = pred['task_id']

            
            if pred.get('status') == 'evaluated' and 'compilation_status' in pred and 'test_status' in pred:
                skipped_count += 1
                print(f"[{idx}/{total_predictions}] Skipping (already evaluated): fragment_id={f_id}, task_id={t_id}")

                
                # if f_id not in results_aggregator:
                #     results_aggregator[f_id] = {'compilations': [], 'tests': []}

                if f_id not in results_aggregator:
                    results_aggregator[f_id] = {'compilations': [], 'tests': [], 'ccn': [], 'nloc': [], 'mi': []}
                
                results_aggregator[f_id]['compilations'].append(pred['compilation_status'])
                results_aggregator[f_id]['tests'].append(pred['test_status'])
                continue
            
            print(f"[{idx}/{total_predictions}] Evaluating: fragment_id={f_id}, task_id={t_id}")
            
            if f_id not in results_aggregator:
                results_aggregator[f_id] = {'compilations': [], 'tests': [], 'ccn': [], 'nloc': [], 'mi': []}

            test_code = dataset.get(t_id, {}).get('test', "")
            
            comp_status, test_status = self.run_java_test(pred['predicted_code'], test_code, f_id, t_id)

            if comp_status == 1:
                q_metrics = self.quality_evaluator.evaluate(pred['predicted_code'])
                results_aggregator[f_id]['ccn'].append(q_metrics['ccn'])
                results_aggregator[f_id]['nloc'].append(q_metrics['nloc'])
                results_aggregator[f_id]['mi'].append(q_metrics['mi'])
            
            pred['status'] = 'evaluated'
            pred['compilation_status'] = comp_status
            pred['test_status'] = test_status
            
            print(f"  → Result: Compilation={'✓' if comp_status else '✗'}, Test={'✓' if test_status else '✗'}")
            
            results_aggregator[f_id]['compilations'].append(comp_status)
            results_aggregator[f_id]['tests'].append(test_status)

            
            if idx % 50 == 0 or idx == total_predictions:
                self._save_predictions(predictions_path, predictions)

        for f_id, scores in results_aggregator.items():
            comp_sum = sum(scores['compilations'])
            test_avg = sum(scores['tests']) / len(scores['tests']) if scores['tests'] else 0

            avg_ccn = sum(scores['ccn']) / len(scores['ccn']) if scores['ccn'] else 0.0
            avg_nloc = sum(scores['nloc']) / len(scores['nloc']) if scores['nloc'] else 0.0
            avg_mi = sum(scores['mi']) / len(scores['mi']) if scores['mi'] else 0.0

            self.literature_jh.update_item(f_id, {
                "status": "evaluated",
                "metadata": {
                    "metrics": {
                        "compilation": comp_sum,
                        "test_pass_rate": test_avg,
                        "ccn": avg_ccn,
                        "nloc": avg_nloc,
                        "mi": avg_mi,
                        "code_bleu": None
                    }
                }
            })
            
            # self.literature_jh.update_item(f_id, {
            #     "status": "evaluated",
            #     "metadata": {
            #         "metrics": {
            #             "compilation": comp_sum,
            #             "test_pass_rate": test_avg,
            #             "code_bleu": None
            #         }
            #     }
            # })
        
        self._save_predictions(predictions_path, predictions)

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        
        print(f"\nEvaluation complete: {skipped_count} skipped, {total_predictions - skipped_count} evaluated")