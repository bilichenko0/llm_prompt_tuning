"""
Stratified train/validation split based on solution_ccn distribution.

Reads statistics/metrics.json and produces:
  - statistics/split_train.json  (10 tasks)
  - statistics/split_val.json    (100 tasks)
  - statistics/split_backup.json (remaining tasks)

The split is done in two stages:
  1. Separate train from the remaining records
  2. Split the remaining into validation and backup

Each output file contains the list of task_ids assigned to that split.

Usage:
    python stratified_split.py [ratio]

    ratio - fraction of data assigned to the validation split
            (default: 0.988, i.e. 10/864 train/remaining)
"""

import json
import sys
from pathlib import Path
from collections import Counter

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    print("[!] sklearn not installed. Install with: pip install scikit-learn")
    sys.exit(1)


def bin_ccn(v: int) -> str:
    if v <= 1:
        return "trivial"
    elif v <= 3:
        return "simple"
    elif v <= 6:
        return "moderate"
    elif v <= 10:
        return "complex"
    else:
        return "hard"


def make_stratum(record_metrics: dict) -> str:
    """Create a stratum label from solution_ccn."""
    return bin_ccn(record_metrics["solution_ccn"])


def main():
    ratio = 0.988
    if len(sys.argv) > 1:
        try:
            ratio = float(sys.argv[1])
            if not 0.0 < ratio < 1.0:
                raise ValueError
        except ValueError:
            print("[!] ratio must be a float between 0 and 1 (exclusive).")
            sys.exit(1)

    project_root = Path(__file__).parent.parent.parent
    metrics_path = project_root / "statistics" / "metrics.json"
    output_dir = project_root / "statistics"

    if not metrics_path.exists():
        print(f"[!] Metrics file not found: {metrics_path}")
        print("    Run compute_metrics.py first.")
        sys.exit(1)

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # Create strata based on ccn
    strata = []
    for m in metrics:
        stratum = make_stratum(m["metrics"])
        strata.append(stratum)

    stratum_counts = Counter(strata)
    total_strata = len(stratum_counts)
    small_strata = sum(1 for c in stratum_counts.values() if c < 5)
    small_records = sum(c for c in stratum_counts.values() if c < 5)

    print(f"Split ratio (val): {ratio}")
    print(f"Total records: {len(strata)}")
    print(f"Non-empty strata: {total_strata}")
    print(f"Strata with < 5 samples: {small_strata} ({small_records} records)")

    # Strategy: keep strata with >= 5 samples for stratified split
    # Records in small strata are added to train set only
    keep_strata = {s for s, c in stratum_counts.items() if c >= 5}
    train_only_indices = [i for i, s in enumerate(strata) if s not in keep_strata]
    splittable_indices = [i for i, s in enumerate(strata) if s in keep_strata]

    print(f"Splittable records: {len(splittable_indices)}")
    print(f"Train-only (small stratum) records: {len(train_only_indices)}")

    # Stage 1: Perform stratified split on splittable records
    splittable_strata = [strata[i] for i in splittable_indices]

    train_idx, val_idx = train_test_split(
        splittable_indices,
        test_size=ratio,
        stratify=splittable_strata,
        random_state=42,
    )

    # Add small-stratum records to train
    train_idx = list(train_idx) + train_only_indices
    val_idx = list(val_idx)

    train_tasks = [metrics[i]["task_id"] for i in train_idx]
    remaining_tasks = [metrics[i]["task_id"] for i in val_idx]

    # Stage 2: Split remaining into validation (100) and backup (rest)
    task_to_metrics = {m["task_id"]: m["metrics"] for m in metrics}
    remaining_strata = [make_stratum(task_to_metrics[tid]) for tid in remaining_tasks]

    desired_val = 100
    backup_ratio = (len(remaining_tasks) - desired_val) / len(remaining_tasks)

    val_split_idx, backup_idx = train_test_split(
        range(len(remaining_tasks)),
        test_size=backup_ratio,
        stratify=remaining_strata,
        random_state=42,
    )

    val_tasks = [remaining_tasks[i] for i in val_split_idx]
    backup_tasks = [remaining_tasks[i] for i in backup_idx]

    # Verify stratification quality for stage 1
    train_strata = Counter(strata[i] for i in train_idx)
    val_strata_counts = Counter(remaining_strata)

    # Verify stratification quality for stage 2
    val_final_strata = Counter(remaining_strata[i] for i in val_split_idx)
    backup_strata_counts = Counter(remaining_strata[i] for i in backup_idx)

    print(f"\n=== Stage 1 Split Results ===")
    print(f"Train: {len(train_tasks)} records ({len(train_strata)} strata)")
    print(f"Remaining: {len(remaining_tasks)} records ({len(val_strata_counts)} strata)")

    print(f"\n=== Stage 2 Split Results ===")
    print(f"Val:     {len(val_tasks)} records ({len(val_final_strata)} strata)")
    print(f"Backup:  {len(backup_tasks)} records ({len(backup_strata_counts)} strata)")

    # Save splits
    with open(output_dir / "split_train.json", "w", encoding="utf-8") as f:
        json.dump(train_tasks, f, indent=2, ensure_ascii=False)

    with open(output_dir / "split_val.json", "w", encoding="utf-8") as f:
        json.dump(val_tasks, f, indent=2, ensure_ascii=False)

    with open(output_dir / "split_backup.json", "w", encoding="utf-8") as f:
        json.dump(backup_tasks, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: statistics/split_train.json ({len(train_tasks)} tasks)")
    print(f"Saved: statistics/split_val.json ({len(val_tasks)} tasks)")
    print(f"Saved: statistics/split_backup.json ({len(backup_tasks)} tasks)")


if __name__ == "__main__":
    main()
