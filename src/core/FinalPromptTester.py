import json
from Predictor import Predictor
from Evaluator import Evaluator
from JSONHandler import JSONHandler
from DirectoryManager import DirectoryManager

import shutil

class FinalPromptTester:
    def __init__(self, predictor: Predictor, evaluator: Evaluator, fragments_jh: JSONHandler, dm: DirectoryManager):
        self.predictor = predictor
        self.evaluator = evaluator
        self.fragments_jh = fragments_jh
        self.dm = dm

    def get_mega_context(self, knapsack_results_path: str) -> str:
        with open(knapsack_results_path, 'r', encoding='utf-8') as f:
            k_data = json.load(f)
        
        selected_ids = [item['id'] for item in k_data.get('selected_fragments', [])]
        
        mega_text = ""
        for f_id in selected_ids:
            frag = self.fragments_jh.get_by_id(f_id)
            if frag:
                mega_text += frag['text'] + "\n\n"
        return mega_text.strip()

    def build_baseline_prompt(self, task_prompt: str) -> str:
        return f"""You are an expert Java developer.
Complete the following Java code:
{task_prompt}"""

    def run_ab_test(self, knapsack_results_path: str, output_path: str):
        mega_context = self.get_mega_context(knapsack_results_path)
        tasks = self.predictor.dataset_sample
        
        results = {
            "baseline": [],
            "mega_prompt": []
        }

        for task in tasks:
            t_id = task['task_id']
            test_code = task.get('test', '')

            baseline_prompt = self.build_baseline_prompt(task['prompt'])
            baseline_raw = self.predictor.get_llm_prediction(baseline_prompt)
            baseline_code = self.predictor.clean_llm_output(baseline_raw) if baseline_raw else ""
            base_comp, base_test = self.evaluator.run_java_test(baseline_code, test_code)
            
            results["baseline"].append({
                "task_id": t_id,
                "compilation": base_comp,
                "test_pass": base_test
            })


            mega_prompt = self.predictor.build_prompt(mega_context, task['prompt'])
            mega_raw = self.predictor.get_llm_prediction(mega_prompt)
            mega_code = self.predictor.clean_llm_output(mega_raw) if mega_raw else ""
            mega_comp, mega_test = self.evaluator.run_java_test(mega_code, test_code)

            results["mega_prompt"].append({
                "task_id": t_id,
                "compilation": mega_comp,
                "test_pass": mega_test
            })

        def aggregate(res_list):
            total = len(res_list)
            comp = sum(x['compilation'] for x in res_list)
            tests = sum(x['test_pass'] for x in res_list)
            return {"total_tasks": total, "successful_compilations": comp, "passed_tests": tests}

        final_report = {
            "summary_baseline": aggregate(results["baseline"]),
            "summary_mega_prompt": aggregate(results["mega_prompt"]),
            "details": results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)
        
        if self.evaluator.temp_dir.exists():
            shutil.rmtree(self.evaluator.temp_dir)
            print(f"--- Temp directory {self.evaluator.temp_dir} deleted. ---")

        return final_report