# import sys
# from pathlib import Path
# import json
# from dotenv import load_dotenv
# import re

# core_path = str(Path(__file__).parent.parent / "core")
# if core_path not in sys.path:
#     sys.path.append(core_path)

# from DirectoryManager import DirectoryManager
# from JSONHandler import JSONHandler
# from SafeGuard import SafeGuard
# from Predictor import Predictor
# from Evaluator import Evaluator
# from FinalPromptTester import FinalPromptTester

# def has_required_imports(code: str) -> bool:
#     """
#     Check if the code has both required imports at the beginning.
    
#     Args:
#         code: The Java code to check
        
#     Returns:
#         True if both imports are present, False otherwise
#     """
#     lines = code.strip().split('\n')
    
#     # Check if the first few lines contain the required imports
#     has_util_import = False
#     has_lang_import = False
    
#     for line in lines[:10]:  # Check first 10 lines to be safe
#         stripped = line.strip()
#         if 'import java.util.*;' in stripped:
#             has_util_import = True
#         if 'import java.lang.*;' in stripped:
#             has_lang_import = True
    
#     return has_util_import and has_lang_import


# def prepend_imports(code: str) -> str:
#     """
#     Prepend the required imports to the code if they're missing.
#     Preserves the class wrapper and other code structure.
    
#     Args:
#         code: The Java code
        
#     Returns:
#         Code with imports prepended if necessary
#     """
#     lines = code.split('\n')
    
#     # Find where to insert imports (before 'class' declaration)
#     class_line_idx = -1
#     existing_imports = []
    
#     for i, line in enumerate(lines):
#         stripped = line.strip()
#         if stripped.startswith('class '):
#             class_line_idx = i
#             break
#         if stripped.startswith('import '):
#             existing_imports.append(i)
    
#     # Remove existing java.util and java.lang imports
#     lines_to_keep = []
#     for i, line in enumerate(lines):
#         stripped = line.strip()
#         # Skip the exact import lines we're going to add
#         if stripped == 'import java.util.*;' or stripped == 'import java.lang.*;':
#             continue
#         lines_to_keep.append(line)
    
#     # Find the new class line index after removing imports
#     class_line_idx = -1
#     for i, line in enumerate(lines_to_keep):
#         if line.strip().startswith('class '):
#             class_line_idx = i
#             break
    
#     # Insert the required imports before the class declaration
#     if class_line_idx >= 0:
#         # Insert imports before class
#         result_lines = (
#             lines_to_keep[:class_line_idx] +
#             ['import java.util.*;', 'import java.lang.*;', ''] +
#             lines_to_keep[class_line_idx:]
#         )
#     else:
#         # No class found, prepend to beginning
#         result_lines = ['import java.util.*;', 'import java.lang.*;', ''] + lines_to_keep
    
#     # Clean up multiple consecutive empty lines
#     cleaned_lines = []
#     prev_empty = False
#     for line in result_lines:
#         is_empty = line.strip() == ''
#         if is_empty and prev_empty:
#             continue
#         cleaned_lines.append(line)
#         prev_empty = is_empty
    
#     return '\n'.join(cleaned_lines).strip()


# def is_wrapped_in_class(code: str) -> bool:
#     """
#     Check if the code contains a class declaration.
    
#     Args:
#         code: The Java code to check
        
#     Returns:
#         True if code contains a class declaration, False otherwise
#     """
#     lines = code.strip().split('\n')
    
#     for line in lines:
#         stripped = line.strip()
#         # Check for class declaration
#         if stripped.startswith('class ') or stripped.startswith('public class '):
#             return True
    
#     return False


# def wrap_in_class(code: str) -> str:
#     """
#     Wrap the code in a Solution class if it's not already wrapped.
    
#     Args:
#         code: The Java code
        
#     Returns:
#         Code wrapped in a class
#     """
#     lines = code.split('\n')
    
#     # Separate imports from the rest of the code
#     import_lines = []
#     code_lines = []
    
#     for line in lines:
#         stripped = line.strip()
#         if stripped.startswith('import '):
#             import_lines.append(line)
#         elif stripped == '':
#             # Keep empty lines with imports if they come before code
#             if not code_lines:
#                 import_lines.append(line)
#             else:
#                 code_lines.append(line)
#         else:
#             code_lines.append(line)
    
#     # Remove trailing empty lines from imports
#     while import_lines and import_lines[-1].strip() == '':
#         import_lines.pop()
    
#     # Remove leading empty lines from code
#     while code_lines and code_lines[0].strip() == '':
#         code_lines.pop(0)
    
#     # Indent the code lines
#     indented_code = []
#     for line in code_lines:
#         if line.strip():  # Only indent non-empty lines
#             indented_code.append('    ' + line)
#         else:
#             indented_code.append(line)
    
#     # Build the wrapped code
#     result = []
    
#     # Add imports
#     if import_lines:
#         result.extend(import_lines)
#         result.append('')
    
#     # Add class wrapper
#     result.append('class Solution {')
#     result.extend(indented_code)
#     result.append('}')
    
#     return '\n'.join(result)


# def prepare_code_for_compilation(code: str) -> str:
#     """
#     Prepare code for compilation by wrapping in class and adding imports if needed.
    
#     Args:
#         code: The raw Java code from LLM
        
#     Returns:
#         Code ready for compilation
#     """
#     # Step 1: Wrap in class if needed
#     if not is_wrapped_in_class(code):
#         code = wrap_in_class(code)
    
#     # Step 2: Add imports if needed
#     if not has_required_imports(code):
#         code = prepend_imports(code)
    
#     return code

# def main():
#     dm = DirectoryManager()
    
#     env_path = dm.get_config_path(".env")
#     load_dotenv(env_path)

#     settings_path = dm.get_config_path("settings.json")
#     try:
#         with open(settings_path, 'r', encoding='utf-8') as f:
#             settings = json.load(f)
#             config = settings.get("llm_config", {})
#             CURRENT_PROVIDER = config.get("provider", "openai")
#             CURRENT_MODEL = config.get("model_name", "gpt-3.5-turbo")
#     except Exception as e:
#         print(f"{e}")
#         CURRENT_PROVIDER = "openai"
#         CURRENT_MODEL = "gpt-3.5-turbo"

#     output_path = dm.get_output_path("effective_java_data.json")
#     dataset_path = dm.get_dataset_path("humaneval.jsonl")
#     knapsack_results_path = dm.get_output_path("knapsack_results.json")
#     final_report_path = dm.get_output_path("final_ab_test_report.json")

#     jh = JSONHandler()
#     jh.load(output_path)
    
#     sg = SafeGuard(jh)
#     #CURRENT_PROVIDER = "openai"

#     predictor = Predictor(jh, sg, dataset_path, provider=CURRENT_PROVIDER, model_name = CURRENT_MODEL)
#     evaluator = Evaluator(jh, dm)

#     # Pass the code preparation function to the tester
#     tester = FinalPromptTester(predictor, evaluator, jh, dm, code_preparer=prepare_code_for_compilation)
    
#     print("Starting A/B test: Baseline vs Mega-Prompt...")
#     print("Step 6: Code will be automatically wrapped in class and imports added if needed for successful compilation")
#     report = tester.run_ab_test(knapsack_results_path, final_report_path)
    
#     print("\n--- Final Results ---")
#     print("Baseline (No Context):")
#     print(f"  Compilations: {report['summary_baseline']['successful_compilations']}/{report['summary_baseline']['total_tasks']}")
#     print(f"  Passed Tests: {report['summary_baseline']['passed_tests']}/{report['summary_baseline']['total_tasks']}")
    
#     print("\nMega-Prompt (With Knapsack Context):")
#     print(f"  Compilations: {report['summary_mega_prompt']['successful_compilations']}/{report['summary_mega_prompt']['total_tasks']}")
#     print(f"  Passed Tests: {report['summary_mega_prompt']['passed_tests']}/{report['summary_mega_prompt']['total_tasks']}")
    
#     print(f"\nDetailed report saved to: {final_report_path}")

# if __name__ == "__main__":
#     main()



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
                knapsack_results_filename = f"knapsack_{book['id']}_{strategy}_{limit}.json"
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

                    print(f"\n==================================================")
                    print(f"A/B Test: {book['id']} ({strategy}) | Limit: {limit} | Model: {model_id}")
                    print(f"==================================================")

                    predictor = Predictor(jh, sg, dataset_path, provider=provider, model_name=model_name, sample_limit=sample_limit)
                    evaluator = Evaluator(jh, cfg.dm)
                    
                    tester = FinalPromptTester(predictor, evaluator, jh, cfg.dm, code_preparer=prepare_code_for_compilation)
                    
                    report = tester.run_ab_test(knapsack_results_path, final_report_path)
                    
                    print(f"\nResults for {model_id} (context limit {limit}):")
                    print(f"  Baseline: {report['summary_baseline']['passed_tests']}/{report['summary_baseline']['total_tasks']} tests")
                    print(f"  Mega-Prompt: {report['summary_mega_prompt']['passed_tests']}/{report['summary_mega_prompt']['total_tasks']} tests")
                    print(f"Report saved to: {final_report_filename}")


if __name__ == "__main__":
    main()