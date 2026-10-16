"""
Script to check all predictions and wrap methods in a class where they are not already wrapped.
"""

import json
import os
import shutil
import re


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


def fix_class_wrapping(predictions_file: str, output_file: str = None):
    """
    Check all predictions and wrap methods in a class where needed.
    
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
    wrapped_count = 0
    already_wrapped = 0
    
    # Process each prediction
    for i, prediction in enumerate(predictions):
        if 'predicted_code' not in prediction:
            print(f"Warning: Prediction {i} has no 'predicted_code' field")
            continue
        
        code = prediction['predicted_code']
        
        if not is_wrapped_in_class(code):
            # Wrap the code in a class
            prediction['predicted_code'] = wrap_in_class(code)
            wrapped_count += 1
            print(f"Wrapped prediction {i} (task_id: {prediction.get('task_id', 'unknown')})")
        else:
            already_wrapped += 1
    
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
    print(f"Already wrapped in class: {already_wrapped}")
    print(f"Wrapped in class: {wrapped_count}")
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
    backup_file = predictions_file.replace('.json', '_backup_wrap.json')
    print(f"Creating backup at: {backup_file}")
    shutil.copy2(predictions_file, backup_file)
    
    # Fix class wrapping
    fix_class_wrapping(predictions_file)
    
    print(f"\nDone! Backup saved to: {backup_file}")


if __name__ == "__main__":
    main()
