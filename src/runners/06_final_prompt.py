import sys
from pathlib import Path
import json
from dotenv import load_dotenv

core_path = str(Path(__file__).parent.parent / "core")
if core_path not in sys.path:
    sys.path.append(core_path)

from DirectoryManager import DirectoryManager
from JSONHandler import JSONHandler
from SafeGuard import SafeGuard
from Predictor import Predictor
from Evaluator import Evaluator
from FinalPromptTester import FinalPromptTester

def main():
    dm = DirectoryManager()
    
    env_path = dm.get_config_path(".env")
    load_dotenv(env_path)

    settings_path = dm.get_config_path("settings.json")
    try:
        with open(settings_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)
            config = settings.get("llm_config", {})
            CURRENT_PROVIDER = config.get("provider", "openai")
            CURRENT_MODEL = config.get("model_name", "gpt-3.5-turbo")
    except Exception as e:
        print(f"{e}")
        CURRENT_PROVIDER = "openai"
        CURRENT_MODEL = "gpt-3.5-turbo"

    output_path = dm.get_output_path("effective_java_data.json")
    dataset_path = dm.get_dataset_path("humaneval.jsonl")
    knapsack_results_path = dm.get_output_path("knapsack_results.json")
    final_report_path = dm.get_output_path("final_ab_test_report.json")

    jh = JSONHandler()
    jh.load(output_path)
    
    sg = SafeGuard(jh)
    #CURRENT_PROVIDER = "openai"

    predictor = Predictor(jh, sg, dataset_path, provider=CURRENT_PROVIDER, model_name = CURRENT_MODEL)
    evaluator = Evaluator(jh, dm)

    tester = FinalPromptTester(predictor, evaluator, jh, dm)
    
    print("Starting A/B test: Baseline vs Mega-Prompt...")
    report = tester.run_ab_test(knapsack_results_path, final_report_path)
    
    print("\n--- Final Results ---")
    print("Baseline (No Context):")
    print(f"  Compilations: {report['summary_baseline']['successful_compilations']}/{report['summary_baseline']['total_tasks']}")
    print(f"  Passed Tests: {report['summary_baseline']['passed_tests']}/{report['summary_baseline']['total_tasks']}")
    
    print("\nMega-Prompt (With Knapsack Context):")
    print(f"  Compilations: {report['summary_mega_prompt']['successful_compilations']}/{report['summary_mega_prompt']['total_tasks']}")
    print(f"  Passed Tests: {report['summary_mega_prompt']['passed_tests']}/{report['summary_mega_prompt']['total_tasks']}")
    
    print(f"\nDetailed report saved to: {final_report_path}")

if __name__ == "__main__":
    main()