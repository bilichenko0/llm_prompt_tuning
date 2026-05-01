import subprocess
import os
import json
from pathlib import Path
import shutil
from JSONHandler import JSONHandler

class Evaluator:
    def __init__(self, literature_jh, directory_manager):
        self.literature_jh = literature_jh
        self.dm = directory_manager
        
        self.temp_dir = Path(self.dm.output_dir) / "temp_java"
        self.temp_dir.mkdir(exist_ok=True)

    def _load_dataset_as_dict(self, dataset_path):
        dataset = {}
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                dataset[item['task_id']] = item
        return dataset

    def run_java_test(self, solution_code, test_code) -> tuple:
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
                print(f"--- Compilation Error ---\n{compile_res.stderr}")
                return (0, 0)

            run_res = subprocess.run(
                ['java', 'Main'],
                cwd=self.temp_dir, capture_output=True, text=True, timeout=10
            )
            return (1, 1 if run_res.returncode == 0 else 0)

        except Exception as e:
            print(f"Execution error: {e}")
            return 0
        # finally:
        #     for f in self.temp_dir.glob("*.class"): f.unlink()

    def evaluate_all(self, predictions_path: str, dataset_path: str):
        with open(predictions_path, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        
        dataset = self._load_dataset_as_dict(dataset_path)
        
        results_aggregator = {}

        for pred in predictions:
            f_id = pred['fragment_id']
            t_id = pred['task_id']
            
            if f_id not in results_aggregator:
                results_aggregator[f_id] = {'compilations': [], 'tests': []}

            test_code = dataset.get(t_id, {}).get('test', "")
            
            comp_status, test_status = self.run_java_test(pred['predicted_code'], test_code)
            pred['status'] = 'evaluated'
            
            results_aggregator[f_id]['compilations'].append(comp_status)
            results_aggregator[f_id]['tests'].append(test_status)

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
        
            with open(predictions_path, 'w', encoding='utf-8') as f:
                json.dump(predictions, f, indent=2, ensure_ascii=False)

            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)