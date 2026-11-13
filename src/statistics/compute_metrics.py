"""
Compute code quality metrics for MBJP dataset records.

Metrics computed per record:
  1. solution_ccn  - Cyclomatic complexity of canonical solution
  2. solution_nloc - Lines of code (NLOC) of canonical solution
  3. test_ccn      - Cyclomatic complexity of test code
  4. test_nloc     - Lines of code (NLOC) of test code
  5. domain        - Problem domain classification (from description keywords)
  6. test_count    - Number of test cases (count of compare() calls)
"""

import json
import os
import sys
import hashlib
from pathlib import Path
from datetime import datetime

import lizard


def compute_ccn_nloc(code: str, entry_point: str = "solve") -> tuple[int, int]:
    """
    Compute cyclomatic complexity and NLOC for a code snippet.

    The snippet is a bare method body, so it must be wrapped in a
    method declaration and class before lizard can analyze it.
    """
    if not code or not code.strip():
        return 1, 0

    safe_name = "".join(ch for ch in entry_point if ch.isalnum() or ch == "_") or "solve"
    wrapped = (
        f"class Solution {{\n"
        f"    public void {safe_name}() {{\n"
        f"{code}\n"
        f"    }}\n"
        f"}}"
    )

    try:
        analysis = lizard.analyze_file.analyze_source_code("Solution.java", wrapped)
        if analysis.function_list:
            func = analysis.function_list[0]
            return func.cyclomatic_complexity, func.nloc
        return 1, 0
    except Exception:
        return 1, 0


def classify_domain(description: str) -> str:
    """
    Classify problem domain based on description keyword analysis.

    Returns one of: list_string, number_math, sorting, array_grid,
    binary, string_processing, or other.
    """
    desc = description.lower()

    if "sort" in desc:
        return "sorting"
    if "binary" in desc:
        return "binary"
    if "array" in desc:
        return "array_grid"
    if "list" in desc or "string" in desc:
        return "list_string"
    if any(kw in desc for kw in ("number", "sum", "count", "math", "digit")):
        return "number_math"
    return "other"


def count_tests(test_code: str) -> int:
    """Count the number of test cases by counting compare() calls."""
    return test_code.count("compare(")


def generate_record_id(task_id: str) -> str:
    """Generate a stable hash ID from task_id (mirrors PDFParser approach)."""
    hash_part = hashlib.md5(task_id.encode()).hexdigest()[:8]
    return f"metric_{hash_part}"


def compute_record_metrics(record: dict) -> dict:
    """Compute all 6 metrics for a single dataset record."""
    canonical = record.get("canonical_solution") or ""
    test = record.get("test") or ""
    entry_point = record.get("entry_point", "solve")
    description = record.get("description", "")

    solution_ccn, solution_nloc = compute_ccn_nloc(canonical, entry_point)
    test_ccn, test_nloc = compute_ccn_nloc(test, "main")
    domain = classify_domain(description)
    test_count = count_tests(test)

    return {
        "id": generate_record_id(record.get("task_id", "")),
        "task_id": record.get("task_id"),
        "entry_point": entry_point,
        "metrics": {
            "solution_ccn": solution_ccn,
            "solution_nloc": solution_nloc,
            "test_ccn": test_ccn,
            "test_nloc": test_nloc,
            "domain": domain,
            "test_count": test_count,
        },
        "_computed_at": datetime.now().isoformat(),
    }


def main():
    project_root = Path(__file__).parent.parent.parent

    dataset_path = project_root / "dataset" / "mbjp_release_v1.2_clean.jsonl"
    if not dataset_path.exists():
        print(f"[!] Dataset not found: {dataset_path}")
        print("    Run clean_dataset.py first or check the path.")
        sys.exit(1)

    output_dir = project_root / "statistics"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "metrics.json"

    print(f"Loading dataset: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"Processing {len(records)} records...")
    results = []
    for i, record in enumerate(records, 1):
        metrics = compute_record_metrics(record)
        results.append(metrics)
        if i % 100 == 0 or i == len(records):
            print(f"  {i}/{len(records)} records processed")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nMetrics saved to: {output_path}")
    print(f"Total records: {len(results)}")

    # Print summary statistics
    ccns = [r["metrics"]["solution_ccn"] for r in results]
    nolcs = [r["metrics"]["solution_nloc"] for r in results]
    domains = [r["metrics"]["domain"] for r in results]
    counts = [r["metrics"]["test_count"] for r in results]

    from collections import Counter
    print(f"\n=== Summary ===")
    print(f"Solution CCN: min={min(ccns)} max={max(ccns)} mean={sum(ccns)/len(ccns):.1f}")
    print(f"Solution NLOC: min={min(nolcs)} max={max(nolcs)} mean={sum(nolcs)/len(nolcs):.1f}")
    print(f"Domains: {Counter(domains)}")
    print(f"Test counts: {Counter(counts)}")


if __name__ == "__main__":
    main()
