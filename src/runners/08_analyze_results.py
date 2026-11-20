import json
import math
from pathlib import Path
from collections import Counter, defaultdict

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
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
        # Chi-square with Edwards continuity correction (clamped at 0)
        chi2_cc = (max(0.0, abs(only_a - only_b) - 1.0) ** 2) / n_discordant
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


def pretty_model_name(model_id: str) -> str:
    mapping = {
        "minimax": "MiniMax-M2.5",
        "mimo_2_6": "MiMo-2.6-Flash",
        "qwen_3_8": "Qwen-3.8-27B",
        "deepseek_4_1": "DeepSeek-V4.1",
        "gemma_4": "Gemma4-31B",
        "mistral": "Mistral-Small-3.2",
        "laguna_2_1": "Laguna-S-2.1",
    }
    return mapping.get(model_id, model_id)


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

    def to_pct(counts: Counter, total: int) -> dict:
        if total == 0:
            return {}
        return {k: round(v / total * 100.0, 2) for k, v in sorted(counts.items())}

    def summarize_subset(task_ids: set[str] | None):
        records = [
            v for k, v in task_meta.items()
            if task_ids is None or k in task_ids
        ]
        n = len(records)
        domains_cnt = Counter(r["domain"] for r in records)
        strata_cnt = Counter(r["stratum"] for r in records)
        return {
            "count": n,
            "solution_ccn": safe_stats([r["solution_ccn"] for r in records]),
            "solution_nloc": safe_stats([r["solution_nloc"] for r in records]),
            "test_ccn": safe_stats([r["test_ccn"] for r in records]),
            "test_nloc": safe_stats([r["test_nloc"] for r in records]),
            "test_count": dict(Counter(r["test_count"] for r in records)),
            "domains": dict(domains_cnt),
            "domains_pct": to_pct(domains_cnt, n),
            "strata": dict(strata_cnt),
            "strata_pct": to_pct(strata_cnt, n),
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


def analyze_model_screening(processed_dir: Path) -> list[dict]:
    screen_files = sorted(processed_dir.glob("prompt_screen_train_*.json"))
    models = []
    for sfile in screen_files:
        if sfile.name.endswith("_summary.json"):
            continue
        data = load_json_safe(sfile)
        if not data or not isinstance(data, dict):
            continue
        summary = data.get("summary")
        if summary:
            models.append(summary)

    models.sort(
        key=lambda x: (x.get("test_pass_rate", 0.0), x.get("compilation_rate", 0.0)),
        reverse=True,
    )
    return models


def analyze_books_and_fragments(processed_dir: Path, cfg: dict) -> dict:
    books_cfg = cfg.get("literature", {}).get("books", [])
    strategies = list(cfg.get("literature", {}).get("strategies", ["paragraph"]))
    if "sentence" not in strategies:
        strategies.append("sentence")

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
                top_fragments = [
                    {
                        "id": item["id"],
                        "tokens_cl100k": item.get("metadata", {}).get("tokens", {}).get("cl100k_base"),
                        "metrics": item.get("metadata", {}).get("metrics", {}),
                        "text_preview": item.get("text", "")[:350],
                    }
                    for item in sorted_eval[:15]
                ]

                # Collect configured single paragraphs from both legacy and new settings format
                id_lookup = {x["id"]: x for x in data}
                configured_sp = {}
                pc_cfg = cfg.get("prompt_comparison", {})
                for conf_key, conf_val in pc_cfg.get("configurations", {}).items():
                    sp_id = conf_val.get("best_single_paragraph", {}).get("fragment_id")
                    if sp_id:
                        sp_item = id_lookup.get(sp_id)
                        configured_sp[conf_key] = {
                            "id": sp_id,
                            "found": sp_item is not None,
                            "tokens": sp_item.get("metadata", {}).get("tokens", {}) if sp_item else None,
                            "metrics": sp_item.get("metadata", {}).get("metrics", {}) if sp_item else None,
                            "full_text": sp_item.get("text", "") if sp_item else None,
                        }

                effective_java_details = {
                    "raw_pass_rates": [pr for pr in pass_rates if pr is not None],
                    "raw_mis": [m for m in mis if m is not None],
                    "raw_ccns": [c for c in ccns if c is not None],
                    "raw_nlocs": [nl for nl in nlocs if nl is not None],
                    "top_15_fragments": top_fragments,
                    "best_single_paragraph_configured": configured_sp,
                    "id_lookup": id_lookup,
                }

    return {
        "books_summary": books_summary,
        "effective_java_details": effective_java_details,
    }


def analyze_predictions_telemetry(processed_dir: Path) -> tuple[dict, dict]:
    pred_files = sorted(processed_dir.glob("predictions_*.json"))
    telemetry = {}
    raw_times_by_model = {}

    for pfile in pred_files:
        preds = load_json_safe(pfile)
        if not preds or not isinstance(preds, list):
            continue

        # Infer model_id from filename: predictions_{book}_{strategy}_{dataset}_{model_id}.json
        stem = pfile.stem
        model_label = stem
        for marker in ("_mbjp_", "_humaneval_"):
            if marker in stem:
                model_label = stem.split(marker, 1)[1]
                break

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
                    times.append(float(meta["time_seconds"]))
                if meta.get("prompt_tokens_api") is not None:
                    prompt_toks_api.append(meta["prompt_tokens_api"])
                if meta.get("completion_tokens_api") is not None:
                    comp_toks_api.append(meta["completion_tokens_api"])
                if meta.get("total_tokens_api") is not None:
                    total_toks_api.append(meta["total_tokens_api"])
                if meta.get("finish_reason"):
                    finish_reasons[meta["finish_reason"]] += 1

        if len(times) >= 50:
            raw_times_by_model[model_label] = times

        telemetry[pfile.name] = {
            "model_id": model_label,
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

    return telemetry, raw_times_by_model


def analyze_knapsack_and_step06(processed_dir: Path, id_lookup: dict, cfg: dict) -> dict:
    knapsack_files = sorted(processed_dir.glob("knapsack_*.json"))
    active_dataset = cfg.get("datasets", {}).get("active_dataset", "mbjp")

    knapsack_table = []
    optimal_fragments_by_limit = {}
    models_found = set()

    for kfile in knapsack_files:
        kdata = load_json_safe(kfile)
        if not kdata or not isinstance(kdata, dict):
            continue

        book = kdata.get("book", "effective_java")
        strat = kdata.get("strategy", "paragraph")
        limit = kdata.get("token_limit")

        # Find ALL Step 06 reports for this (book, strategy, limit) across all models
        prefix = f"final_ab_test_report_{book}_{strat}_{limit}_{active_dataset}_"
        ab_candidates = sorted(processed_dir.glob(f"{prefix}*.json"))

        step06_by_model = {}
        for ab_file in ab_candidates:
            ab_data = load_json_safe(ab_file)
            if not ab_data:
                continue
            model_id = ab_file.stem[len(prefix):] if ab_file.stem.startswith(prefix) else ab_file.stem
            models_found.add(model_id)

            base_s = ab_data.get("summary_baseline", {})
            mega_s = ab_data.get("summary_mega_prompt", {})
            base_tot = max(1, base_s.get("total_tasks", 1))
            mega_tot = max(1, mega_s.get("total_tasks", 1))

            step06_by_model[model_id] = {
                "report_file": ab_file.name,
                "baseline": {
                    **base_s,
                    "compilation_rate": round(base_s.get("successful_compilations", 0) / base_tot, 4),
                    "test_pass_rate": round(base_s.get("passed_tests", 0) / base_tot, 4),
                },
                "mega_prompt": {
                    **mega_s,
                    "compilation_rate": round(mega_s.get("successful_compilations", 0) / mega_tot, 4),
                    "test_pass_rate": round(mega_s.get("passed_tests", 0) / mega_tot, 4),
                },
                "details": ab_data.get("details"),
            }

        selected = kdata.get("selected_fragments", [])
        knapsack_table.append({
            "file": kfile.name,
            "book": book,
            "strategy": strat,
            "token_limit": limit,
            "total_selected": int(kdata.get("total_selected", len(selected))),
            "total_tokens": kdata.get("total_tokens"),
            "total_value": round(kdata.get("total_value", 0.0), 4),
            "selected_ids": [s.get("id") for s in selected],
            "step06_by_model": step06_by_model,
        })

        if book == "effective_java" and strat == "paragraph" and limit in (700, 1000):
            optimal_fragments_by_limit[str(limit)] = [
                {
                    "id": s.get("id"),
                    "weight_metric": s.get("weight_metric"),
                    "tokens": s.get("tokens"),
                    "full_text": id_lookup.get(s.get("id"), {}).get("text", ""),
                    "metrics": id_lookup.get(s.get("id"), {}).get("metadata", {}).get("metrics", {}),
                }
                for s in selected
            ]

    knapsack_table.sort(key=lambda x: (x["book"], x["strategy"], x["token_limit"] or 0))
    return {
        "models_in_step06": sorted(models_found),
        "knapsack_runs": knapsack_table,
        "optimal_fragments_details_by_limit": optimal_fragments_by_limit,
    }


def analyze_step07_validation(processed_dir: Path, task_meta: dict) -> dict:
    pc_files = sorted(processed_dir.glob("prompt_comparison_*.json"))
    comparisons = {}
    by_model_tasks = {}

    approaches = ["simple_baseline", "best_single_paragraph", "best_optimal_prompt"]

    for pc_file in pc_files:
        pc_data = load_json_safe(pc_file)
        if not pc_data or not isinstance(pc_data, dict):
            continue

        model_id = pc_data.get("model_id", pc_file.stem)
        details = pc_data.get("details", {})
        by_task = defaultdict(dict)

        for app in approaches:
            for row in details.get(app, []):
                tid = row["task_id"]
                by_task[tid][app] = {
                    "compilation": int(row.get("compilation", 0)),
                    "test_pass": int(row.get("test_pass", 0)),
                }

        by_model_tasks[model_id] = by_task
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

        summary_enriched = {}
        for app, stats in pc_data.get("summary", {}).items():
            tot = max(1, stats.get("total_tasks", 1))
            summary_enriched[app] = {
                **stats,
                "compilation_rate": round(stats.get("successful_compilations", 0) / tot, 4),
                "test_pass_rate": round(stats.get("passed_tests", 0) / tot, 4),
            }

        comparisons[pc_file.name] = {
            "model_id": model_id,
            "dataset": pc_data.get("dataset"),
            "book": pc_data.get("book", pc_data.get("optimal_prompt", {}).get("book", "effective_java")),
            "split": pc_data.get("split"),
            "single_paragraph_config": pc_data.get("single_paragraph"),
            "optimal_prompt_config": pc_data.get("optimal_prompt"),
            "summary": summary_enriched,
            "domain_breakdown": dict(domain_breakdown),
            "stratum_breakdown": dict(stratum_breakdown),
            "mcnemar_statistical_tests": mcnemar_results,
            "divergent_tasks_count": len(task_differences),
            "divergent_tasks": task_differences,
        }

    # Cross-model McNemar tests on common validation tasks
    cross_model = {}
    model_ids = sorted(by_model_tasks.keys())
    for i in range(len(model_ids)):
        for j in range(i + 1, len(model_ids)):
            m1, m2 = model_ids[i], model_ids[j]
            common_tids = sorted(set(by_model_tasks[m1].keys()) & set(by_model_tasks[m2].keys()))
            if not common_tids:
                continue
            pair_res = {"common_tasks": len(common_tids)}
            for app in approaches:
                v1_pass = [by_model_tasks[m1][t].get(app, {}).get("test_pass", 0) for t in common_tids]
                v2_pass = [by_model_tasks[m2][t].get(app, {}).get("test_pass", 0) for t in common_tids]
                v1_comp = [by_model_tasks[m1][t].get(app, {}).get("compilation", 0) for t in common_tids]
                v2_comp = [by_model_tasks[m2][t].get(app, {}).get("compilation", 0) for t in common_tids]
                pair_res[app] = {
                    "test_pass": mcnemar_test(v1_pass, v2_pass),
                    "compilation": mcnemar_test(v1_comp, v2_comp),
                }
            cross_model[f"{m1}_vs_{m2}"] = pair_res

    return {
        "per_model": comparisons,
        "cross_model_mcnemar": cross_model,
    }


def _plot_knapsack_value_count_ax(ax, limits, total_values, selected_counts):
    y_max = int(math.ceil(max(max(selected_counts, default=1), max(total_values, default=1)))) + 1
    ticks = list(range(0, y_max + 1))

    bars = ax.bar(
        limits,
        selected_counts,
        width=46,
        alpha=0.25,
        color="#C44E52",
        edgecolor="#C44E52",
        linewidth=1.0,
        label="Selected Fragments Count",
        zorder=2,
    )
    ax.plot(
        limits,
        total_values,
        marker="o",
        color="#4C72B0",
        linewidth=2.2,
        markersize=6,
        label="Total Knapsack Value",
        zorder=3,
    )

    ax.set_title("Knapsack Optimization by Token Limit")
    ax.set_xlabel("Token Limit (W)")
    ax.set_ylabel("Sum of Selected Fragment Values", color="#4C72B0")
    ax.tick_params(axis="y", labelcolor="#4C72B0")
    ax.set_xticks(limits)
    ax.set_ylim(0, y_max)
    ax.set_yticks(ticks)
    ax.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax.legend(loc="upper left", fontsize=8.5, frameon=True)

    ax_twin = ax.twinx()
    ax_twin.set_ylabel("Selected Fragments Count", color="#C44E52")
    ax_twin.tick_params(axis="y", labelcolor="#C44E52")
    ax_twin.set_ylim(0, y_max)
    ax_twin.set_yticks(ticks)
    ax_twin.grid(False)

    for bar, cnt in zip(bars, selected_counts):
        ax.annotate(
            f"{int(cnt)}",
            xy=(bar.get_x() + bar.get_width() / 2, cnt),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#9E2A2B",
            zorder=4,
        )


def _plot_step06_metric_ax(ax, k_runs, models, metric_key: str, title: str, ylabel: str):
    palette = {
        "minimax": "#4C72B0",
        "mimo_2_6": "#55A868",
    }
    fallback_colors = ["#C44E52", "#8172B2", "#CCB974", "#64B5CD"]

    limits_all = [r["token_limit"] for r in k_runs]

    for idx_m, m_id in enumerate(models):
        color = palette.get(m_id, fallback_colors[idx_m % len(fallback_colors)])
        label_name = pretty_model_name(m_id)

        mega_pts = []
        base_pts = []
        for r in k_runs:
            m_data = r.get("step06_by_model", {}).get(m_id)
            if not m_data:
                continue
            lim = r["token_limit"]
            mega_val = m_data["mega_prompt"].get(metric_key)
            base_val = m_data["baseline"].get(metric_key)
            if mega_val is not None:
                mega_pts.append((lim, mega_val * 100.0))
            if base_val is not None:
                base_pts.append((lim, base_val * 100.0))

        if mega_pts:
            ax.plot(
                [p[0] for p in mega_pts],
                [p[1] for p in mega_pts],
                marker="s",
                linewidth=2,
                color=color,
                label=f"{label_name} (Mega-Prompt)",
            )
        if base_pts:
            ax.plot(
                [p[0] for p in base_pts],
                [p[1] for p in base_pts],
                marker="^",
                linestyle="--",
                linewidth=1.4,
                alpha=0.75,
                color=color,
                label=f"{label_name} (Baseline)",
            )

    ax.set_title(title)
    ax.set_xlabel("Token Limit (W)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(limits_all)
    ax.set_ylim(0, 108)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)


def generate_plots(
    plots_dir: Path,
    ds_analysis: dict,
    screening_models: list[dict],
    books_analysis: dict,
    knapsack_analysis: dict,
    val_analysis: dict,
    raw_times_by_model: dict,
):
    if not HAS_MATPLOTLIB:
        return

    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

    # ---------------------------------------------------------
    # 1. Dataset Stratification Verification Plot
    # ---------------------------------------------------------
    ds_sum = ds_analysis.get("summary", {})
    if ds_sum.get("all", {}).get("count", 0) > 0:
        strata_order = ["trivial", "simple", "moderate", "complex", "hard"]
        domains_order = sorted(ds_sum["all"]["domains_pct"].keys())
        splits = [("all", "Full MBJP"), ("train", "Train Split"), ("val", "Validation Split")]
        colors = ["#4C72B0", "#DD8452", "#55A868"]

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        w = 0.25

        xs = range(len(strata_order))
        for idx_s, (sk, slbl) in enumerate(splits):
            vals = [ds_sum.get(sk, {}).get("strata_pct", {}).get(st, 0.0) for st in strata_order]
            n_sub = ds_sum.get(sk, {}).get("count", 0)
            axes[0].bar([i + (idx_s - 1) * w for i in xs], vals, w, label=f"{slbl} (n={n_sub})", color=colors[idx_s])
        axes[0].set_title("Dataset Stratification by Complexity (CCN Stratum)")
        axes[0].set_xticks(list(xs))
        axes[0].set_xticklabels(strata_order)
        axes[0].set_ylabel("Share of Tasks (%)")
        axes[0].legend()
        axes[0].grid(axis="y", linestyle="--", alpha=0.5)

        xd = range(len(domains_order))
        for idx_s, (sk, slbl) in enumerate(splits):
            vals = [ds_sum.get(sk, {}).get("domains_pct", {}).get(dm, 0.0) for dm in domains_order]
            n_sub = ds_sum.get(sk, {}).get("count", 0)
            axes[1].bar([i + (idx_s - 1) * w for i in xd], vals, w, label=f"{slbl} (n={n_sub})", color=colors[idx_s])
        axes[1].set_title("Dataset Stratification by Task Domain")
        axes[1].set_xticks(list(xd))
        axes[1].set_xticklabels(domains_order, rotation=20)
        axes[1].set_ylabel("Share of Tasks (%)")
        axes[1].legend()
        axes[1].grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(plots_dir / "fig_dataset_stratification.png")
        plt.close(fig)

    # ---------------------------------------------------------
    # 2. Model Screening Plot (Step 00)
    # ---------------------------------------------------------
    if screening_models:
        labels = [pretty_model_name(m["model_id"]) for m in screening_models]
        comp_r = [m.get("compilation_rate", 0.0) * 100.0 for m in screening_models]
        pass_r = [m.get("test_pass_rate", 0.0) * 100.0 for m in screening_models]

        x = range(len(labels))
        w = 0.35
        fig, ax = plt.subplots(figsize=(9, 4.8))
        r1 = ax.bar([i - w / 2 for i in x], comp_r, w, label="Compilation Rate (%)", color="#4C72B0")
        r2 = ax.bar([i + w / 2 for i in x], pass_r, w, label="Test Pass Rate (%)", color="#55A868")
        ax.set_title("Initial Model Screening on Train Split (Simple Baseline)")
        ax.set_ylabel("Rate (%)")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=15)
        ax.set_ylim(0, 112)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        for rect in list(r1) + list(r2):
            h = rect.get_height()
            ax.annotate(f"{h:.0f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
        plt.tight_layout()
        plt.savefig(plots_dir / "fig_screening_models.png")
        plt.close(fig)

    # ---------------------------------------------------------
    # 3. Fragment Evaluation Distributions (Stage 04: Pass Rate, MI, CCN, NLOC)
    # ---------------------------------------------------------
    ej_details = books_analysis.get("effective_java_details", {})
    pass_rates = ej_details.get("raw_pass_rates", [])
    mis = ej_details.get("raw_mis", [])
    ccns = ej_details.get("raw_ccns", [])
    nlocs = ej_details.get("raw_nlocs", [])

    if pass_rates:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

        axes[0, 0].hist(pass_rates, bins=11, range=(0, 1.0), color="#4C72B0", edgecolor="black", alpha=0.85)
        axes[0, 0].set_title("Test Pass Rate Distribution (Effective Java)")
        axes[0, 0].set_xlabel("Test Pass Rate")
        axes[0, 0].set_ylabel("Number of Fragments")
        axes[0, 0].grid(axis="y", linestyle="--", alpha=0.5)

        if mis:
            axes[0, 1].hist(mis, bins=15, color="#55A868", edgecolor="black", alpha=0.85)
            axes[0, 1].set_title("Maintainability Index (MI) Distribution")
            axes[0, 1].set_xlabel("Average MI")
            axes[0, 1].set_ylabel("Number of Fragments")
            axes[0, 1].grid(axis="y", linestyle="--", alpha=0.5)

        if ccns:
            axes[1, 0].hist(ccns, bins=15, color="#DD8452", edgecolor="black", alpha=0.85)
            axes[1, 0].set_title("Cyclomatic Complexity (CCN) Distribution")
            axes[1, 0].set_xlabel("Average CCN")
            axes[1, 0].set_ylabel("Number of Fragments")
            axes[1, 0].grid(axis="y", linestyle="--", alpha=0.5)

        if nlocs:
            axes[1, 1].hist(nlocs, bins=15, color="#8172B2", edgecolor="black", alpha=0.85)
            axes[1, 1].set_title("Lines of Code (NLOC) Distribution")
            axes[1, 1].set_xlabel("Average NLOC")
            axes[1, 1].set_ylabel("Number of Fragments")
            axes[1, 1].grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(plots_dir / "fig_fragment_evaluation_dist.png")
        plt.close(fig)

    # ---------------------------------------------------------
    # 4. Knapsack Scaling (Standalone Plots + Combined 1x3 Figure)
    # ---------------------------------------------------------
    k_runs = [
        r for r in knapsack_analysis.get("knapsack_runs", [])
        if r.get("book") == "effective_java" and r.get("strategy") == "paragraph"
    ]
    step06_models = knapsack_analysis.get("models_in_step06", [])

    if k_runs:
        limits = [r["token_limit"] for r in k_runs]
        selected_counts = [int(r["total_selected"]) for r in k_runs]
        total_values = [r["total_value"] for r in k_runs]

        # Value & Integer Fragment Count
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        _plot_knapsack_value_count_ax(ax, limits, total_values, selected_counts)
        plt.tight_layout()
        plt.savefig(plots_dir / "fig_knapsack_value_count.png")
        plt.close(fig)

        # Test Pass Rate across token limits (all models)
        if step06_models:
            fig, ax = plt.subplots(figsize=(7.2, 4.6))
            _plot_step06_metric_ax(
                ax, k_runs, step06_models,
                metric_key="test_pass_rate",
                title="Train Split Test Pass Rate by Token Limit",
                ylabel="Test Pass Rate (%)",
            )
            plt.tight_layout()
            plt.savefig(plots_dir / "fig_knapsack_test_pass_rate.png")
            plt.close(fig)

            # Compilation Rate across token limits (all models)
            fig, ax = plt.subplots(figsize=(7.2, 4.6))
            _plot_step06_metric_ax(
                ax, k_runs, step06_models,
                metric_key="compilation_rate",
                title="Train Split Compilation Rate by Token Limit",
                ylabel="Compilation Rate (%)",
            )
            plt.tight_layout()
            plt.savefig(plots_dir / "fig_knapsack_compilation_rate.png")
            plt.close(fig)

        
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
        _plot_knapsack_value_count_ax(axes[0], limits, total_values, selected_counts)
        if step06_models:
            _plot_step06_metric_ax(
                axes[1], k_runs, step06_models,
                metric_key="test_pass_rate",
                title="Train Test Pass Rate by Limit",
                ylabel="Test Pass Rate (%)",
            )
            _plot_step06_metric_ax(
                axes[2], k_runs, step06_models,
                metric_key="compilation_rate",
                title="Train Compilation Rate by Limit",
                ylabel="Compilation Rate (%)",
            )
        plt.tight_layout()
        plt.savefig(plots_dir / "fig_knapsack_scaling.png")
        plt.close(fig)

    # ---------------------------------------------------------
    # 5. Stage 03 Inference Time Violin Plot (minimax vs mimo_2_6)
    # ---------------------------------------------------------
    if raw_times_by_model:
        ordered_models = sorted(raw_times_by_model.keys())
        data_vecs = [raw_times_by_model[m] for m in ordered_models]
        labels = [f"{pretty_model_name(m)}\n(n={len(raw_times_by_model[m])})" for m in ordered_models]

        fig, ax = plt.subplots(figsize=(7.5, 5))
        parts = ax.violinplot(data_vecs, showmeans=True, showmedians=True, showextrema=True)
        violin_colors = ["#4C72B0", "#55A868", "#DD8452", "#8172B2"]
        for idx_b, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(violin_colors[idx_b % len(violin_colors)])
            pc.set_edgecolor("black")
            pc.set_alpha(0.65)
        parts["cmeans"].set_color("#C44E52")
        parts["cmedians"].set_color("black")

        ax.set_title("Stage 03 LLM Inference Time Distribution by Model")
        ax.set_ylabel("Inference Time (seconds)")
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        for idx_m, vec in enumerate(data_vecs, 1):
            st = safe_stats(vec)
            ax.annotate(
                f"median={st['median']:.2f}s\nmean={st['mean']:.2f}s",
                xy=(idx_m, st["median"]),
                xytext=(12, 8),
                textcoords="offset points",
                fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85),
            )

        plt.tight_layout()
        plt.savefig(plots_dir / "fig_stage03_inference_time_violin.png")
        plt.close(fig)

    # ---------------------------------------------------------
    # 6. Step 07 Validation Plots (Per-Model + Cross-Model Comparison)
    # ---------------------------------------------------------
    per_model_val = val_analysis.get("per_model", {})
    labels = ["Simple Baseline", "Best Single Paragraph", "Optimal Prompt (Knapsack)"]
    keys = ["simple_baseline", "best_single_paragraph", "best_optimal_prompt"]

    if per_model_val:
        val_entries = sorted(per_model_val.values(), key=lambda x: x.get("model_id", ""))

        # Individual per-model overall & breakdown plots
        for idx_v, vdata in enumerate(val_entries):
            m_id = vdata.get("model_id", f"model_{idx_v}")
            m_pretty = pretty_model_name(m_id)
            summary = vdata.get("summary", {})
            n_tasks = summary.get("simple_baseline", {}).get("total_tasks", 100)

            comp_rates = [summary.get(k, {}).get("compilation_rate", 0.0) * 100.0 for k in keys]
            pass_rates_val = [summary.get(k, {}).get("test_pass_rate", 0.0) * 100.0 for k in keys]

            x = range(len(labels))
            width = 0.35

            fig, ax = plt.subplots(figsize=(8, 5))
            rects1 = ax.bar([i - width / 2 for i in x], comp_rates, width, label="Compilation Success (%)", color="#4C72B0")
            rects2 = ax.bar([i + width / 2 for i in x], pass_rates_val, width, label="Test Pass Rate (%)", color="#55A868")
            ax.set_ylabel("Success Rate (%)")
            ax.set_title(f"{m_pretty}: Validation Comparison ({n_tasks} MBJP Tasks)")
            ax.set_xticks(list(x))
            ax.set_xticklabels(labels)
            ax.set_ylim(0, 112)
            ax.legend()
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            for rect in list(rects1) + list(rects2):
                h = rect.get_height()
                ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                            textcoords="offset points", ha="center", va="bottom", fontsize=9)
            plt.tight_layout()
            plt.savefig(plots_dir / f"fig_val_comparison_overall_{m_id}.png")
            if len(val_entries) == 1:
                plt.savefig(plots_dir / "fig_val_comparison_overall.png")
            plt.close(fig)

            # Breakdown by CCN Stratum and Domain
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
                axes[0].set_title(f"{m_pretty}: Pass Rate by Complexity (CCN)")
                axes[0].set_xticks(list(xs))
                axes[0].set_xticklabels([f"{s}\n(n={vdata['stratum_breakdown'][s]['simple_baseline']['total']})" for s in strata_order])
                axes[0].set_ylabel("Test Pass Rate (%)")
                axes[0].set_ylim(0, 112)
                axes[0].legend(fontsize=8.5)
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
                axes[1].set_title(f"{m_pretty}: Pass Rate by Domain")
                axes[1].set_xticks(list(xd))
                axes[1].set_xticklabels(
                    [f"{d}\n(n={vdata['domain_breakdown'][d]['simple_baseline']['total']})" for d in domains_order],
                    rotation=20,
                )
                axes[1].set_ylabel("Test Pass Rate (%)")
                axes[1].set_ylim(0, 112)
                axes[1].legend(fontsize=8.5)
                axes[1].grid(axis="y", linestyle="--", alpha=0.5)

            plt.tight_layout()
            plt.savefig(plots_dir / f"fig_val_comparison_breakdowns_{m_id}.png")
            if idx_v == 0:
                plt.savefig(plots_dir / "fig_val_comparison_breakdowns.png")
            plt.close(fig)

        # Multi-model comparison plot (minimax vs mimo_2_6 side-by-side)
        if len(val_entries) >= 2:
            fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
            model_colors = ["#4C72B0", "#55A868", "#DD8452", "#8172B2"]
            n_models = len(val_entries)
            w = 0.7 / n_models
            x = range(len(labels))

            for metric_idx, (m_key, m_title, m_ylabel) in enumerate([
                ("test_pass_rate", "Validation Test Pass Rate: Model Comparison", "Test Pass Rate (%)"),
                ("compilation_rate", "Validation Compilation Rate: Model Comparison", "Compilation Success (%)"),
            ]):
                ax = axes[metric_idx]
                for idx_m, vdata in enumerate(val_entries):
                    m_id = vdata.get("model_id", f"model_{idx_m}")
                    vals = [vdata.get("summary", {}).get(k, {}).get(m_key, 0.0) * 100.0 for k in keys]
                    offset = (idx_m - (n_models - 1) / 2.0) * w
                    rects = ax.bar(
                        [i + offset for i in x],
                        vals,
                        w,
                        label=pretty_model_name(m_id),
                        color=model_colors[idx_m % len(model_colors)],
                    )
                    for rect in rects:
                        h = rect.get_height()
                        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

                ax.set_title(m_title)
                ax.set_ylabel(m_ylabel)
                ax.set_xticks(list(x))
                ax.set_xticklabels(labels)
                ax.set_ylim(0, 112)
                ax.legend()
                ax.grid(axis="y", linestyle="--", alpha=0.5)

            plt.tight_layout()
            plt.savefig(plots_dir / "fig_models_comparison_val.png")
            plt.savefig(plots_dir / "fig_val_comparison_overall.png")
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

    print("2. Analyzing Step 00 model screening...")
    screening_models = analyze_model_screening(processed_dir)

    print("3. Analyzing parsed books and evaluated fragments...")
    books_analysis = analyze_books_and_fragments(processed_dir, cfg)

    print("4. Analyzing Stage 03 LLM prediction telemetry...")
    telemetry_analysis, raw_times_by_model = analyze_predictions_telemetry(processed_dir)

    print("5. Analyzing Knapsack runs and Step 06 reports (all models)...")
    id_lookup = books_analysis.get("effective_java_details", {}).get("id_lookup", {})
    knapsack_analysis = analyze_knapsack_and_step06(processed_dir, id_lookup, cfg)

    print("6. Analyzing Step 07 validation prompt comparison & McNemar tests (all models)...")
    val_analysis = analyze_step07_validation(processed_dir, ds_analysis["task_meta"])

    print("7. Generating publication plots in statistics/plots/...")
    generate_plots(
        plots_dir,
        ds_analysis,
        screening_models,
        books_analysis,
        knapsack_analysis,
        val_analysis,
        raw_times_by_model,
    )

    ej_clean = dict(books_analysis.get("effective_java_details", {}))
    for k in ("id_lookup", "raw_pass_rates", "raw_mis", "raw_ccns", "raw_nlocs"):
        ej_clean.pop(k, None)

    summary_report = {
        "dataset_and_splits": ds_analysis["summary"],
        "step00_model_screening": screening_models,
        "literature_books_summary": books_analysis["books_summary"],
        "effective_java_highlights": ej_clean,
        "predictions_telemetry": telemetry_analysis,
        "knapsack_and_step06": knapsack_analysis,
        "step07_validation_comparison": val_analysis,
        "notes_on_missing_stage06_07_metrics": (
            "Stage 06 (final_ab_test_report_*.json) and Stage 07 (prompt_comparison_*.json) "
            "only persisted compilation and test_pass flags without predicted_code, MI, CCN, NLOC, "
            "or inference_time. Code quality metrics (MI, CCN, NLOC) and inference_time are therefore "
            "reported from Stage 03-04 fragment evaluation."
        ),
    }

    out_json_path = stats_dir / "thesis_analysis_summary.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Summary JSON saved to: {out_json_path}")
    if HAS_MATPLOTLIB:
        print(f"[OK] Plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()