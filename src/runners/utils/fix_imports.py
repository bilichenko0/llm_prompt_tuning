"""
Script to check all predictions and prepend missing default imports.
The two default imports are:
- import java.util.*;
- import java.lang.*;
"""

import json
import os
import shutil


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


def fix_predictions_imports(predictions_file: str, output_file: str = None):
    """
    Check all predictions and prepend missing imports.
    
    Args:
        predictions_file: Path to the predictions JSON file
        output_file: Optional output file path. If None, overwrites input file.
    """
    print(f"Loading predictions from: {predictions_file}")
    
    # Load predictions directly with json module
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    
    if not isinstance(predictions, list):
        print("Error: Predictions file should contain a list of predictions")
        return
    
    print(f"Found {len(predictions)} predictions")
    
    # Track statistics
    fixed_count = 0
    already_correct = 0
    
    # Process each prediction
    for i, prediction in enumerate(predictions):
        if 'predicted_code' not in prediction:
            print(f"Warning: Prediction {i} has no 'predicted_code' field")
            continue
        
        code = prediction['predicted_code']
        
        if not has_required_imports(code):
            # Fix the code by prepending imports
            prediction['predicted_code'] = prepend_imports(code)
            fixed_count += 1
            print(f"Fixed prediction {i} (task_id: {prediction.get('task_id', 'unknown')})")
        else:
            already_correct += 1
    
    # Save the updated predictions
    if output_file is None:
        output_file = predictions_file
    
    print(f"\nSaving updated predictions to: {output_file}")
    
    # Write directly with json module
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total predictions: {len(predictions)}")
    print(f"Already had imports: {already_correct}")
    print(f"Fixed (imports added): {fixed_count}")
    print("="*60)


def main():
    """Main entry point."""
    # Get the project root directory (two levels up from this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Default predictions file
    predictions_file = os.path.join(project_root, "processed", "predictions.json")
    
    # Check if file exists
    if not os.path.exists(predictions_file):
        print(f"Error: Predictions file not found: {predictions_file}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Script directory: {script_dir}")
        print(f"Project root: {project_root}")
        return
    
    # Create backup using shutil
    backup_file = predictions_file.replace('.json', '_backup.json')
    print(f"Creating backup at: {backup_file}")
    shutil.copy2(predictions_file, backup_file)
    
    # Fix imports
    fix_predictions_imports(predictions_file)
    
    print(f"\nDone! Backup saved to: {backup_file}")


if __name__ == "__main__":
    main()
