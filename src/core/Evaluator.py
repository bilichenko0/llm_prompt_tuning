import subprocess
import os
import json
from pathlib import Path
import shutil
from datetime import datetime
from JSONHandler import JSONHandler

class Evaluator:
    def __init__(self, literature_jh, directory_manager):
        self.literature_jh = literature_jh
        self.dm = directory_manager
        
        self.temp_dir = Path(self.dm.output_dir) / "temp_java"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Initialize error log file
        self.error_log_path = Path(self.dm.output_dir) / "evaluation_errors.log"

    def _load_dataset_as_dict(self, dataset_path):
        dataset = {}
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                dataset[item['task_id']] = item
        return dataset

    def _log_error(self, fragment_id, task_id, error_type, error_details):
        """Log errors to a separate error log file."""
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

    def run_java_test(self, solution_code, test_code, fragment_id=None, task_id=None) -> tuple:
        self.temp_dir.mkdir(exist_ok=True)

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
                
                # Log compilation error
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
            
            # Timeout means the test failed (infinite loop or hung process)
            return (1, 0)  # Compiled successfully but test failed
        except Exception as e:
            error_msg = str(e)
            print(f"--- Execution Error ---\n{error_msg}")
            
            # Log execution error
            if fragment_id and task_id:
                self._log_error(fragment_id, task_id, "execution_error", error_msg)
            
            return (0, 0)

    def evaluate_all(self, predictions_path: str, dataset_path: str):
        # Load predictions from disk
        with open(predictions_path, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        
        dataset = self._load_dataset_as_dict(dataset_path)
        
        # Check for existing evaluation results and resume from there
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
            
            # Skip if already evaluated
            if pred.get('status') == 'evaluated' and 'compilation_status' in pred and 'test_status' in pred:
                skipped_count += 1
                print(f"[{idx}/{total_predictions}] Skipping (already evaluated): fragment_id={f_id}, task_id={t_id}")
                
                # Rebuild results_aggregator from existing data
                if f_id not in results_aggregator:
                    results_aggregator[f_id] = {'compilations': [], 'tests': []}
                results_aggregator[f_id]['compilations'].append(pred['compilation_status'])
                results_aggregator[f_id]['tests'].append(pred['test_status'])
                continue
            
            print(f"[{idx}/{total_predictions}] Evaluating: fragment_id={f_id}, task_id={t_id}")
            
            if f_id not in results_aggregator:
                results_aggregator[f_id] = {'compilations': [], 'tests': []}

            test_code = dataset.get(t_id, {}).get('test', "")
            
            comp_status, test_status = self.run_java_test(pred['predicted_code'], test_code, f_id, t_id)
            
            # Store individual results in the prediction
            pred['status'] = 'evaluated'
            pred['compilation_status'] = comp_status
            pred['test_status'] = test_status
            
            print(f"  → Result: Compilation={'✓' if comp_status else '✗'}, Test={'✓' if test_status else '✗'}")
            
            results_aggregator[f_id]['compilations'].append(comp_status)
            results_aggregator[f_id]['tests'].append(test_status)
            
            # Save to disk after each iteration for robustness
            with open(predictions_path, 'w', encoding='utf-8') as f:
                json.dump(predictions, f, indent=2, ensure_ascii=False)

        # Update literature fragments with aggregated metrics
        for f_id, scores in results_aggregator.items():
            comp_sum = sum(scores['compilations'])
            test_avg = sum(scores['tests']) / len(scores['tests']) if scores['tests'] else 0
            
            self.literature_jh.update_item(f_id, {
                "status": "evaluated",
                "metadata": {
                    "metrics": {
                        "compilation": comp_sum,
                        "test_pass_rate": test_avg,
                        "code_bleu": None
                    }
                }
            })
        
        # Final save
        with open(predictions_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)

        # Cleanup temp directory
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        
        print(f"\nEvaluation complete: {skipped_count} skipped, {total_predictions - skipped_count} evaluated")