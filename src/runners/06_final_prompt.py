import sys
from pathlib import Path
import json
from dotenv import load_dotenv
import re

core_path = str(Path(__file__).parent.parent / "core")
if core_path not in sys.path:
    sys.path.append(core_path)

from DirectoryManager import DirectoryManager
from JSONHandler import JSONHandler
from SafeGuard import SafeGuard
from Predictor import Predictor
from Evaluator import Evaluator
from FinalPromptTester import FinalPromptTester

def has_required_imports(code: str) -> bool:
    """
    Check if the code has both required imports at the beginning.
    
    Args:
        code: The Java code to check
        
    Returns:
        True if both imports are present, False otherwise
    """
    lines = code.strip().split('\n')
    
    # Check if the first few lines contain the required imports
    has_util_import = False
    has_lang_import = False
    
    for line in lines[:10]:  # Check first 10 lines to be safe
        stripped = line.strip()
        if 'import java.util.*;' in stripped:
            has_util_import = True
        if 'import java.lang.*;' in stripped:
            has_lang_import = True
    
    return has_util_import and has_lang_import


def prepend_imports(code: str) -> str:
    """
    Prepend the required imports to the code if they're missing.
    Preserves the class wrapper and other code structure.
    
    Args:
        code: The Java code
        
    Returns:
        Code with imports prepended if necessary
    """
    lines = code.split('\n')
    
    # Find where to insert imports (before 'class' declaration)
    class_line_idx = -1
    existing_imports = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('class '):
            class_line_idx = i
            break
        if stripped.startswith('import '):
            existing_imports.append(i)
    
    # Remove existing java.util and java.lang imports
    lines_to_keep = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Skip the exact import lines we're going to add
        if stripped == 'import java.util.*;' or stripped == 'import java.lang.*;':
            continue
        lines_to_keep.append(line)
    
    # Find the new class line index after removing imports
    class_line_idx = -1
    for i, line in enumerate(lines_to_keep):
        if line.strip().startswith('class '):
            class_line_idx = i
            break
    
    # Insert the required imports before the class declaration
    if class_line_idx >= 0:
        # Insert imports before class
        result_lines = (
            lines_to_keep[:class_line_idx] +
            ['import java.util.*;', 'import java.lang.*;', ''] +
            lines_to_keep[class_line_idx:]
        )
    else:
        # No class found, prepend to beginning
        result_lines = ['import java.util.*;', 'import java.lang.*;', ''] + lines_to_keep
    
    # Clean up multiple consecutive empty lines
    cleaned_lines = []
    prev_empty = False
    for line in result_lines:
        is_empty = line.strip() == ''
        if is_empty and prev_empty:
            continue
        cleaned_lines.append(line)
        prev_empty = is_empty
    
    return '\n'.join(cleaned_lines).strip()


def is_wrapped_in_class(code: str) -> bool:
    """
    Check if the code contains a class declaration.
    
    Args:
        code: The Java code to check
        
    Returns:
        True if code contains a class declaration, False otherwise
    """
    lines = code.strip().split('\n')
    
    for line in lines:
        stripped = line.strip()
        # Check for class declaration
        if stripped.startswith('class ') or stripped.startswith('public class '):
            return True
    
    return False


def wrap_in_class(code: str) -> str:
    """
    Wrap the code in a Solution class if it's not already wrapped.
    
    Args:
        code: The Java code
        
    Returns:
        Code wrapped in a class
    """
    lines = code.split('\n')
    
    # Separate imports from the rest of the code
    import_lines = []
    code_lines = []
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('import '):
            import_lines.append(line)
        elif stripped == '':
            # Keep empty lines with imports if they come before code
            if not code_lines:
                import_lines.append(line)
            else:
                code_lines.append(line)
        else:
            code_lines.append(line)
    
    # Remove trailing empty lines from imports
    while import_lines and import_lines[-1].strip() == '':
        import_lines.pop()
    
    # Remove leading empty lines from code
    while code_lines and code_lines[0].strip() == '':
        code_lines.pop(0)
    
    # Indent the code lines
    indented_code = []
    for line in code_lines:
        if line.strip():  # Only indent non-empty lines
            indented_code.append('    ' + line)
        else:
            indented_code.append(line)
    
    # Build the wrapped code
    result = []
    
    # Add imports
    if import_lines:
        result.extend(import_lines)
        result.append('')
    
    # Add class wrapper
    result.append('class Solution {')
    result.extend(indented_code)
    result.append('}')
    
    return '\n'.join(result)


def prepare_code_for_compilation(code: str) -> str:
    """
    Prepare code for compilation by wrapping in class and adding imports if needed.
    
    Args:
        code: The raw Java code from LLM
        
    Returns:
        Code ready for compilation
    """
    # Step 1: Wrap in class if needed
    if not is_wrapped_in_class(code):
        code = wrap_in_class(code)
    
    # Step 2: Add imports if needed
    if not has_required_imports(code):
        code = prepend_imports(code)
    
    return code

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

    # Pass the code preparation function to the tester
    tester = FinalPromptTester(predictor, evaluator, jh, dm, code_preparer=prepare_code_for_compilation)
    
    print("Starting A/B test: Baseline vs Mega-Prompt...")
    print("Step 6: Code will be automatically wrapped in class and imports added if needed for successful compilation")
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