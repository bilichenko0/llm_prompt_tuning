"""
Stratified train/validation split based on solution_ccn distribution.

Reads statistics/metrics.json and produces:
  - statistics/split_train.json
  - statistics/split_val.json

Each output file contains the list of task_ids assigned to that split.

Usage:
    python stratified_split.py [ratio]

    ratio - fraction of data assigned to the validation split
            (default: 0.988, i.e. 10/864 train/val)
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

    # Perform stratified split on splittable records
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
    val_tasks = [metrics[i]["task_id"] for i in val_idx]

    # Verify stratification quality
    train_strata = Counter(strata[i] for i in train_idx)
    val_strata = Counter(strata[i] for i in val_idx)

    overlap_quality = []
    for s, c in stratum_counts.items():
        if s in keep_strata:
            train_c = train_strata.get(s, 0)
            val_c = val_strata.get(s, 0)
            stratum_val_ratio = val_c / c
            overlap_quality.append(stratum_val_ratio)

    avg_val_ratio = sum(overlap_quality) / len(overlap_quality) if overlap_quality else 0

    print(f"\n=== Split Results ===")
    print(f"Train: {len(train_tasks)} records ({len(train_strata)} strata)")
    print(f"Val:   {len(val_tasks)} records ({len(val_strata)} strata)")
    print(f"Average val ratio (target {ratio}): {avg_val_ratio:.3f}")

    # Check for strata fully in train or val
    train_only_count = sum(1 for s in keep_strata if val_strata.get(s, 0) == 0)
    print(f"Strata missing from val: {train_only_count}")

    # Save splits
    with open(output_dir / "split_train.json", "w", encoding="utf-8") as f:
        json.dump(train_tasks, f, indent=2, ensure_ascii=False)

    with open(output_dir / "split_val.json", "w", encoding="utf-8") as f:
        json.dump(val_tasks, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: statistics/split_train.json ({len(train_tasks)} tasks)")
    print(f"Saved: statistics/split_val.json ({len(val_tasks)} tasks)")


if __name__ == "__main__":
    main()
