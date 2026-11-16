import json
import math
from pathlib import Path
from collections import Counter, defaultdict

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[!] matplotlib is not installed. JSON summary will be generated, but plots will be skipped.")
    print("    To generate plots, run: pip install matplotlib")


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


def safe_stats(values: list[float]) -> dict:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0}
    clean.sort()
    n = len(clean)
    mean_val = sum(clean) / n
    median_val = clean[n // 2] if n % 2 == 1 else (clean[n // 2 - 1] + clean[n // 2]) / 2.0
    variance = sum((x - mean_val) ** 2 for x in clean) / n if n > 0 else 0.0
    return {
        "count": n,
        "min": round(clean[0], 4),
        "max": round(clean[-1], 4),
        "mean": round(mean_val, 4),
        "median": round(median_val, 4),
        "std": round(math.sqrt(variance), 4),
    }


def mcnemar_test(a_pass: list[int], b_pass: list[int]) -> dict:
    both_pass = 0
    only_a = 0
    only_b = 0
    both_fail = 0

    for a, b in zip(a_pass, b_pass):
        if a == 1 and b == 1:
            both_pass += 1
        elif a == 1 and b == 0:
            only_a += 1
        elif a == 0 and b == 1:
            only_b += 1
        else:
            both_fail += 1

    n_discordant = only_a + only_b
    if n_discordant == 0:
        chi2_cc = 0.0
        p_chi2 = 1.0
        p_exact = 1.0
    else:
        # Chi-square with Edwards continuity correction
        chi2_cc = ((abs(only_a - only_b) - 1.0) ** 2) / n_discordant
        # Survival function for chi2(df=1) is erfc(sqrt(x / 2))
        p_chi2 = math.erfc(math.sqrt(chi2_cc / 2.0))

        # Exact two-sided binomial p-value for Binom(n_discordant, 0.5)
        k = min(only_a, only_b)
        cum_prob = sum(math.comb(n_discordant, i) for i in range(k + 1)) / (2 ** n_discordant)
        p_exact = min(1.0, 2.0 * cum_prob)

    return {
        "both_pass": both_pass,
        "only_first_pass": only_a,
        "only_second_pass": only_b,
        "both_fail": both_fail,
        "discordant_pairs": n_discordant,
        "chi2_continuity_corrected": round(chi2_cc, 4),
        "p_value_chi2": round(p_chi2, 6),
        "p_value_exact_binomial": round(p_exact, 6),
        "significant_at_0_05": p_exact < 0.05,
    }


def load_json_safe(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[!] Could not read {path.name}: {e}")
        return None


def analyze_dataset_and_splits(stats_dir: Path) -> dict:
    metrics_data = load_json_safe(stats_dir / "metrics.json") or []
    split_train = set(load_json_safe(stats_dir / "split_train.json") or [])
    split_val = set(load_json_safe(stats_dir / "split_val.json") or [])
    split_backup = set(load_json_safe(stats_dir / "split_backup.json") or [])

    task_meta = {}
    for item in metrics_data:
        tid = item.get("task_id")
        m = item.get("metrics", {})
        ccn = m.get("solution_ccn", 1)
        task_meta[tid] = {
            "solution_ccn": ccn,
            "solution_nloc": m.get("solution_nloc", 0),
            "test_ccn": m.get("test_ccn", 0),
            "test_nloc": m.get("test_nloc", 0),
            "domain": m.get("domain", "other"),
            "test_count": m.get("test_count", 0),
            "stratum": bin_ccn(ccn),
        }

    def summarize_subset(task_ids: set[str] | None):
        records = [
            v for k, v in task_meta.items()
            if task_ids is None or k in task_ids
        ]
        return {
            "count": len(records),
            "solution_ccn": safe_stats([r["solution_ccn"] for r in records]),
            "solution_nloc": safe_stats([r["solution_nloc"] for r in records]),
            "test_ccn": safe_stats([r["test_ccn"] for r in records]),
            "test_nloc": safe_stats([r["test_nloc"] for r in records]),
            "test_count": dict(Counter(r["test_count"] for r in records)),
            "domains": dict(Counter(r["domain"] for r in records)),
            "strata": dict(Counter(r["stratum"] for r in records)),
        }

    return {
        "task_meta": task_meta,
        "summary": {
            "all": summarize_subset(None),
            "train": summarize_subset(split_train),
            "val": summarize_subset(split_val),
            "backup": summarize_subset(split_backup),
            "train_task_ids": sorted(split_train),
        },
    }


def analyze_books_and_fragments(processed_dir: Path, cfg: dict) -> dict:
    books_cfg = cfg.get("literature", {}).get("books", [])
    strategies = cfg.get("literature", {}).get("strategies", ["paragraph", "sentence"])
    if "sentence" not in strategies:
        strategies = list(strategies) + ["sentence"]

    books_summary = {}
    effective_java_details = {}

    for b in books_cfg:
        b_id = b["id"]
        for strat in strategies:
            fname = f"{b_id}_{strat}.json"
            fpath = processed_dir / fname
            data = load_json_safe(fpath)
            if not data or not isinstance(data, list):
                continue

            status_counts = Counter(item.get("status") for item in data)
            lengths = [item.get("metadata", {}).get("length", len(item.get("text", ""))) for item in data]
            tok_cl100k = [
                item.get("metadata", {}).get("tokens", {}).get("cl100k_base")
                for item in data
                if isinstance(item.get("metadata", {}).get("tokens"), dict)
            ]
            tok_o200k = [
                item.get("metadata", {}).get("tokens", {}).get("o200k_base")
                for item in data
                if isinstance(item.get("metadata", {}).get("tokens"), dict)
            ]

            evaluated_items = [item for item in data if item.get("status") == "evaluated"]
            pass_rates = [
                item.get("metadata", {}).get("metrics", {}).get("test_pass_rate")
                for item in evaluated_items
            ]
            comp_counts = [
                item.get("metadata", {}).get("metrics", {}).get("compilation")
                for item in evaluated_items
            ]
            ccns = [
                item.get("metadata", {}).get("metrics", {}).get("ccn")
                for item in evaluated_items
                if (item.get("metadata", {}).get("metrics", {}).get("ccn") or 0) > 0
            ]
            nlocs = [
                item.get("metadata", {}).get("metrics", {}).get("nloc")
                for item in evaluated_items
                if (item.get("metadata", {}).get("metrics", {}).get("nloc") or 0) > 0
            ]
            mis = [
                item.get("metadata", {}).get("metrics", {}).get("mi")
                for item in evaluated_items
                if (item.get("metadata", {}).get("metrics", {}).get("mi") or 0) > 0
            ]

            key = f"{b_id}_{strat}"
            books_summary[key] = {
                "book_id": b_id,
                "strategy": strat,
                "total_fragments": len(data),
                "status_counts": dict(status_counts),
                "char_length": safe_stats(lengths),
                "tokens_cl100k_base": safe_stats(tok_cl100k),
                "tokens_o200k_base": safe_stats(tok_o200k),
                "evaluated_count": len(evaluated_items),
                "metrics_summary": {
                    "test_pass_rate": safe_stats(pass_rates),
                    "compilation_sum": safe_stats(comp_counts),
                    "ccn_nonzero": safe_stats(ccns),
                    "nloc_nonzero": safe_stats(nlocs),
                    "mi_nonzero": safe_stats(mis),
                    "pass_rate_distribution": dict(
                        Counter(round(pr, 2) for pr in pass_rates if pr is not None)
                    ),
                },
            }

            if b_id == "effective_java" and strat == "paragraph":
                sorted_eval = sorted(
                    evaluated_items,
                    key=lambda x: (
                        x.get("metadata", {}).get("metrics", {}).get("test_pass_rate") or 0.0,
                        -(x.get("metadata", {}).get("tokens", {}).get("cl100k_base") or 999999),
                    ),
                    reverse=True,
                )
                top_fragments = []
                for item in sorted_eval[:15]:
                    top_fragments.append({
                        "id": item["id"],
                        "tokens_cl100k": item.get("metadata", {}).get("tokens", {}).get("cl100k_base"),
                        "metrics": item.get("metadata", {}).get("metrics", {}),
                        "text_preview": item.get("text", "")[:350],
                    })

                target_sp_id = (
                    cfg.get("prompt_comparison", {})
                    .get("best_single_paragraph", {})
                    .get("fragment_id", "node_1fee7206")
                )
                sp_item = next((x for x in data if x["id"] == target_sp_id), None)

                effective_java_details = {
                    "raw_pass_rates": [pr for pr in pass_rates if pr is not None],
                    "raw_mis": [m for m in mis if m is not None],
                    "top_15_fragments": top_fragments,
                    "best_single_paragraph_ configured": {
                        "id": target_sp_id,
                        "found": sp_item is not None,
                        "tokens": sp_item.get("metadata", {}).get("tokens", {}) if sp_item else None,
                        "metrics": sp_item.get("metadata", {}).get("metrics", {}) if sp_item else None,
                        "full_text": sp_item.get("text", "") if sp_item else None,
                    },
                    "id_lookup": {x["id"]: x for x in data},
                }

    return {
        "books_summary": books_summary,
        "effective_java_details": effective_java_details,
    }


def analyze_predictions_telemetry(processed_dir: Path) -> dict:
    pred_files = sorted(processed_dir.glob("predictions_*.json"))
    telemetry = {}

    for pfile in pred_files:
        preds = load_json_safe(pfile)
        if not preds or not isinstance(preds, list):
            continue

        times = []
        prompt_toks_api = []
        comp_toks_api = []
        total_toks_api = []
        finish_reasons = Counter()
        comp_statuses = []
        test_statuses = []
        per_task_pass = defaultdict(list)
        per_task_comp = defaultdict(list)

        for p in preds:
            if "compilation_status" in p:
                comp_statuses.append(p["compilation_status"])
                per_task_comp[p["task_id"]].append(p["compilation_status"])
            if "test_status" in p:
                test_statuses.append(p["test_status"])
                per_task_pass[p["task_id"]].append(p["test_status"])

            meta = p.get("llm_metadata") or {}
            if isinstance(meta, dict):
                if meta.get("time_seconds") is not None:
                    times.append(meta["time_seconds"])
                if meta.get("prompt_tokens_api") is not None:
                    prompt_toks_api.append(meta["prompt_tokens_api"])
                if meta.get("completion_tokens_api") is not None:
                    comp_toks_api.append(meta["completion_tokens_api"])
                if meta.get("total_tokens_api") is not None:
                    total_toks_api.append(meta["total_tokens_api"])
                if meta.get("finish_reason"):
                    finish_reasons[meta["finish_reason"]] += 1

        telemetry[pfile.name] = {
            "total_predictions": len(preds),
            "overall_compilation_rate": round(sum(comp_statuses) / len(comp_statuses), 4) if comp_statuses else None,
            "overall_test_pass_rate": round(sum(test_statuses) / len(test_statuses), 4) if test_statuses else None,
            "time_seconds": safe_stats(times),
            "prompt_tokens_api": safe_stats(prompt_toks_api),
            "completion_tokens_api": safe_stats(comp_toks_api),
            "total_tokens_api": safe_stats(total_toks_api),
            "finish_reasons": dict(finish_reasons),
            "per_train_task_summary": {
                tid: {
                    "evaluations": len(per_task_pass[tid]),
                    "compilation_rate": round(sum(per_task_comp[tid]) / len(per_task_comp[tid]), 4) if per_task_comp[tid] else 0.0,
                    "test_pass_rate": round(sum(per_task_pass[tid]) / len(per_task_pass[tid]), 4) if per_task_pass[tid] else 0.0,
                }
                for tid in sorted(per_task_pass.keys())
            },
        }

    return telemetry


def analyze_knapsack_and_step06(processed_dir: Path, id_lookup: dict) -> dict:
    knapsack_files = sorted(processed_dir.glob("knapsack_*.json"))
    knapsack_table = []
    optimal_700_fragments = []

    for kfile in knapsack_files:
        kdata = load_json_safe(kfile)
        if not kdata or not isinstance(kdata, dict):
            continue

        book = kdata.get("book", "effective_java")
        strat = kdata.get("strategy", "paragraph")
        limit = kdata.get("token_limit")

        ab_candidates = list(processed_dir.glob(f"final_ab_test_report_{book}_{strat}_{limit}_*.json"))
        ab_summary = None
        if ab_candidates:
            ab_data = load_json_safe(ab_candidates[0])
            if ab_data:
                ab_summary = {
                    "report_file": ab_candidates[0].name,
                    "baseline": ab_data.get("summary_baseline"),
                    "mega_prompt": ab_data.get("summary_mega_prompt"),
                    "details": ab_data.get("details"),
                }

        selected = kdata.get("selected_fragments", [])
        knapsack_table.append({
            "file": kfile.name,
            "book": book,
            "strategy": strat,
            "token_limit": limit,
            "total_selected": kdata.get("total_selected", len(selected)),
            "total_tokens": kdata.get("total_tokens"),
            "total_value": round(kdata.get("total_value", 0.0), 4),
            "selected_ids": [s.get("id") for s in selected],
            "step06_train_ab_test": ab_summary,
        })

        if book == "effective_java" and strat == "paragraph" and limit == 700:
            for s in selected:
                fid = s.get("id")
                full_node = id_lookup.get(fid, {})
                optimal_700_fragments.append({
                    "id": fid,
                    "weight_metric": s.get("weight_metric"),
                    "tokens": s.get("tokens"),
                    "full_text": full_node.get("text", ""),
                    "metrics": full_node.get("metadata", {}).get("metrics", {}),
                })

    knapsack_table.sort(key=lambda x: (x["book"], x["strategy"], x["token_limit"] or 0))
    return {
        "knapsack_runs": knapsack_table,
        "optimal_700_fragments_details": optimal_700_fragments,
    }


def analyze_step07_validation(processed_dir: Path, task_meta: dict) -> dict:
    pc_files = sorted(processed_dir.glob("prompt_comparison_*.json"))
    comparisons = {}

    for pc_file in pc_files:
        pc_data = load_json_safe(pc_file)
        if not pc_data or not isinstance(pc_data, dict):
            continue

        details = pc_data.get("details", {})
        approaches = ["simple_baseline", "best_single_paragraph", "best_optimal_prompt"]
        by_task = defaultdict(dict)

        for app in approaches:
            for row in details.get(app, []):
                tid = row["task_id"]
                by_task[tid][app] = {
                    "compilation": int(row.get("compilation", 0)),
                    "test_pass": int(row.get("test_pass", 0)),
                }

        sorted_tids = sorted(by_task.keys())

        domain_breakdown = defaultdict(lambda: {a: {"total": 0, "comp": 0, "pass": 0} for a in approaches})
        stratum_breakdown = defaultdict(lambda: {a: {"total": 0, "comp": 0, "pass": 0} for a in approaches})

        for tid in sorted_tids:
            meta = task_meta.get(tid, {"domain": "unknown", "stratum": "unknown"})
            dom = meta["domain"]
            strat = meta["stratum"]
            for app in approaches:
                res = by_task[tid].get(app, {"compilation": 0, "test_pass": 0})
                domain_breakdown[dom][app]["total"] += 1
                domain_breakdown[dom][app]["comp"] += res["compilation"]
                domain_breakdown[dom][app]["pass"] += res["test_pass"]

                stratum_breakdown[strat][app]["total"] += 1
                stratum_breakdown[strat][app]["comp"] += res["compilation"]
                stratum_breakdown[strat][app]["pass"] += res["test_pass"]

        pairs = [
            ("simple_baseline", "best_optimal_prompt"),
            ("simple_baseline", "best_single_paragraph"),
            ("best_single_paragraph", "best_optimal_prompt"),
        ]
        mcnemar_results = {}
        for a1, a2 in pairs:
            vec_pass_1 = [by_task[tid].get(a1, {}).get("test_pass", 0) for tid in sorted_tids]
            vec_pass_2 = [by_task[tid].get(a2, {}).get("test_pass", 0) for tid in sorted_tids]
            vec_comp_1 = [by_task[tid].get(a1, {}).get("compilation", 0) for tid in sorted_tids]
            vec_comp_2 = [by_task[tid].get(a2, {}).get("compilation", 0) for tid in sorted_tids]

            mcnemar_results[f"{a1}_vs_{a2}"] = {
                "test_pass": mcnemar_test(vec_pass_1, vec_pass_2),
                "compilation": mcnemar_test(vec_comp_1, vec_comp_2),
            }

        task_differences = []
        for tid in sorted_tids:
            sb = by_task[tid].get("simple_baseline", {}).get("test_pass", 0)
            sp = by_task[tid].get("best_single_paragraph", {}).get("test_pass", 0)
            op = by_task[tid].get("best_optimal_prompt", {}).get("test_pass", 0)
            if len({sb, sp, op}) > 1:
                meta = task_meta.get(tid, {})
                task_differences.append({
                    "task_id": tid,
                    "domain": meta.get("domain"),
                    "solution_ccn": meta.get("solution_ccn"),
                    "stratum": meta.get("stratum"),
                    "simple_baseline_pass": sb,
                    "best_single_paragraph_pass": sp,
                    "best_optimal_prompt_pass": op,
                })

        comparisons[pc_file.name] = {
            "model_id": pc_data.get("model_id"),
            "dataset": pc_data.get("dataset"),
            "split": pc_data.get("split"),
            "single_paragraph_config": pc_data.get("single_paragraph"),
            "optimal_prompt_config": pc_data.get("optimal_prompt"),
            "summary": pc_data.get("summary"),
            "domain_breakdown": dict(domain_breakdown),
            "stratum_breakdown": dict(stratum_breakdown),
            "mcnemar_statistical_tests": mcnemar_results,
            "divergent_tasks_count": len(task_differences),
            "divergent_tasks": task_differences,
        }

    return comparisons


def generate_plots(plots_dir: Path, books_analysis: dict, knapsack_analysis: dict, val_analysis: dict):
    if not HAS_MATPLOTLIB:
        return

    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

    ej_details = books_analysis.get("effective_java_details", {})
    pass_rates = ej_details.get("raw_pass_rates", [])
    mis = ej_details.get("raw_mis", [])

    if pass_rates:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].hist(pass_rates, bins=11, range=(0, 1.0), color="#4C72B0", edgecolor="black", alpha=0.85)
        axes[0].set_title("Test Pass Rate Distribution (Effective Java)")
        axes[0].set_xlabel("Test Pass Rate")
        axes[0].set_ylabel("Number of Fragments (Paragraphs)")
        axes[0].grid(axis="y", linestyle="--", alpha=0.5)

        if mis:
            axes[1].hist(mis, bins=15, color="#55A868", edgecolor="black", alpha=0.85)
            axes[1].set_title("Code Maintainability Index (MI)")
            axes[1].set_xlabel("Average Maintainability Index (MI)")
            axes[1].set_ylabel("Number of Fragments")
            axes[1].grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(plots_dir / "fig_fragment_evaluation_dist.png")
        plt.close(fig)

    k_runs = [
        r for r in knapsack_analysis.get("knapsack_runs", [])
        if r.get("book") == "effective_java" and r.get("strategy") == "paragraph"
    ]
    if k_runs:
        limits = [r["token_limit"] for r in k_runs]
        selected_counts = [r["total_selected"] for r in k_runs]
        total_values = [r["total_value"] for r in k_runs]
        train_mega_pass = [
            (r["step06_train_ab_test"]["mega_prompt"]["passed_tests"]
             / r["step06_train_ab_test"]["mega_prompt"]["total_tasks"] * 100.0)
            if r.get("step06_train_ab_test") and r["step06_train_ab_test"].get("mega_prompt")
            else None
            for r in k_runs
        ]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        ax1 = axes[0]
        ax1.plot(limits, total_values, marker="o", color="#4C72B0", linewidth=2, label="Total Knapsack Value")
        ax1.set_title("Knapsack Optimization by Token Limit")
        ax1.set_xlabel("Token Limit (W)")
        ax1.set_ylabel("Sum of Selected Fragment Values", color="#4C72B0")
        ax1.grid(True, linestyle="--", alpha=0.5)

        ax1_twin = ax1.twinx()
        ax1_twin.bar(limits, selected_counts, width=40, alpha=0.25, color="#C44E52", label="Fragment Count")
        ax1_twin.set_ylabel("Selected Fragments Count", color="#C44E52")

        ax2 = axes[1]
        if any(v is not None for v in train_mega_pass):
            valid_pts = [(l, v) for l, v in zip(limits, train_mega_pass) if v is not None]
            ax2.plot(
                [p[0] for p in valid_pts],
                [p[1] for p in valid_pts],
                marker="s",
                color="#55A868",
                linewidth=2,
                label="Mega-Prompt (Train split)",
            )
            base_vals = [
                (r["step06_train_ab_test"]["baseline"]["passed_tests"]
                 / r["step06_train_ab_test"]["baseline"]["total_tasks"] * 100.0)
                for r in k_runs
                if r.get("step06_train_ab_test") and r["step06_train_ab_test"].get("baseline")
            ]
            if base_vals:
                ax2.plot(
                    [p[0] for p in valid_pts],
                    base_vals,
                    marker="^",
                    linestyle="--",
                    color="#8172B2",
                    linewidth=1.5,
                    label="Baseline (Train split)",
                )
            ax2.set_title("Train Split Pass Rate by Token Limit")
            ax2.set_xlabel("Token Limit (W)")
            ax2.set_ylabel("Test Pass Rate (%)")
            ax2.set_ylim(0, 105)
            ax2.legend()
            ax2.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(plots_dir / "fig_knapsack_scaling.png")
        plt.close(fig)

    if val_analysis:
        first_key = next(iter(val_analysis))
        vdata = val_analysis[first_key]
        summary = vdata.get("summary", {})

        labels = ["Simple Baseline", "Best Single Paragraph", "Optimal Prompt (Knapsack)"]
        keys = ["simple_baseline", "best_single_paragraph", "best_optimal_prompt"]

        comp_rates = [
            summary.get(k, {}).get("successful_compilations", 0)
            / max(1, summary.get(k, {}).get("total_tasks", 100)) * 100.0
            for k in keys
        ]
        pass_rates_val = [
            summary.get(k, {}).get("passed_tests", 0)
            / max(1, summary.get(k, {}).get("total_tasks", 100)) * 100.0
            for k in keys
        ]

        x = range(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        rects1 = ax.bar([i - width / 2 for i in x], comp_rates, width, label="Compilation Success (%)", color="#4C72B0")
        rects2 = ax.bar([i + width / 2 for i in x], pass_rates_val, width, label="Test Pass Rate (%)", color="#55A868")

        ax.set_ylabel("Success Rate (%)")
        ax.set_title("Approach Comparison on Validation Split (100 MBJP Tasks)")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 110)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        for rect in list(rects1) + list(rects2):
            h = rect.get_height()
            ax.annotate(
                f"{h:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(plots_dir / "fig_val_comparison_overall.png")
        plt.close(fig)

        strata_order = [s for s in ["trivial", "simple", "moderate", "complex", "hard"] if s in vdata["stratum_breakdown"]]
        domains_order = sorted(vdata["domain_breakdown"].keys())

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        colors = ["#8172B2", "#CCB974", "#55A868"]

        if strata_order:
            w = 0.25
            xs = range(len(strata_order))
            for idx_k, (k, lbl) in enumerate(zip(keys, labels)):
                vals = [
                    vdata["stratum_breakdown"][s][k]["pass"]
                    / max(1, vdata["stratum_breakdown"][s][k]["total"]) * 100.0
                    for s in strata_order
                ]
                axes[0].bar([i + (idx_k - 1) * w for i in xs], vals, w, label=lbl, color=colors[idx_k])
            axes[0].set_title("Test Pass Rate by Task Complexity (CCN)")
            axes[0].set_xticks(list(xs))
            axes[0].set_xticklabels([f"{s}\n(n={vdata['stratum_breakdown'][s]['simple_baseline']['total']})" for s in strata_order])
            axes[0].set_ylabel("Test Pass Rate (%)")
            axes[0].set_ylim(0, 110)
            axes[0].legend()
            axes[0].grid(axis="y", linestyle="--", alpha=0.5)

        if domains_order:
            w = 0.25
            xd = range(len(domains_order))
            for idx_k, (k, lbl) in enumerate(zip(keys, labels)):
                vals = [
                    vdata["domain_breakdown"][d][k]["pass"]
                    / max(1, vdata["domain_breakdown"][d][k]["total"]) * 100.0
                    for d in domains_order
                ]
                axes[1].bar([i + (idx_k - 1) * w for i in xd], vals, w, label=lbl, color=colors[idx_k])
            axes[1].set_title("Test Pass Rate by Task Domain")
            axes[1].set_xticks(list(xd))
            axes[1].set_xticklabels(
                [f"{d}\n(n={vdata['domain_breakdown'][d]['simple_baseline']['total']})" for d in domains_order],
                rotation=20,
            )
            axes[1].set_ylabel("Test Pass Rate (%)")
            axes[1].set_ylim(0, 110)
            axes[1].legend()
            axes[1].grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(plots_dir / "fig_val_comparison_breakdowns.png")
        plt.close(fig)


def main():
    project_root = Path(__file__).parent.parent.parent.absolute()
    configs_dir = project_root / "configs"
    processed_dir = project_root / "processed"
    stats_dir = project_root / "statistics"
    plots_dir = stats_dir / "plots"

    stats_dir.mkdir(exist_ok=True)

    cfg = load_json_safe(configs_dir / "settings.json") or {}

    print("1. Analyzing dataset metrics and stratified splits...")
    ds_analysis = analyze_dataset_and_splits(stats_dir)

    print("2. Analyzing parsed books and evaluated fragments...")
    books_analysis = analyze_books_and_fragments(processed_dir, cfg)

    print("3. Analyzing LLM prediction telemetry...")
    telemetry_analysis = analyze_predictions_telemetry(processed_dir)

    print("4. Analyzing Knapsack runs and Step 06 reports...")
    id_lookup = books_analysis.get("effective_java_details", {}).get("id_lookup", {})
    knapsack_analysis = analyze_knapsack_and_step06(processed_dir, id_lookup)

    print("5. Analyzing Step 07 validation prompt comparison & McNemar tests...")
    val_analysis = analyze_step07_validation(processed_dir, ds_analysis["task_meta"])

    print("6. Generating publication plots in statistics/plots/...")
    generate_plots(plots_dir, books_analysis, knapsack_analysis, val_analysis)

    ej_clean = dict(books_analysis.get("effective_java_details", {}))
    ej_clean.pop("id_lookup", None)
    ej_clean.pop("raw_pass_rates", None)
    ej_clean.pop("raw_mis", None)

    summary_report = {
        "dataset_and_splits": ds_analysis["summary"],
        "literature_books_summary": books_analysis["books_summary"],
        "effective_java_highlights": ej_clean,
        "predictions_telemetry": telemetry_analysis,
        "knapsack_and_step06": knapsack_analysis,
        "step07_validation_comparison": val_analysis,
    }

    out_json_path = stats_dir / "thesis_analysis_summary.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Summary JSON saved to: {out_json_path}")
    if HAS_MATPLOTLIB:
        print(f"[OK] Plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()