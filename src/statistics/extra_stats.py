import json
import math
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, mannwhitneyu

ROOT = Path(__file__).resolve().parent.parent.parent
CORE_DIR = ROOT / "src" / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.append(str(CORE_DIR))

from CodeQualityEvaluator import CodeQualityEvaluator

PROCESSED = ROOT / "processed"
PLOTS = ROOT / "statistics" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

TIMEOUT_THRESHOLD = 120.0


def safe_stats(vals: list[float]) -> dict:
    clean = sorted(float(v) for v in vals if v is not None)
    if not clean:
        return {"n": 0, "min": 0, "max": 0, "mean": 0, "median": 0, "std": 0}
    n = len(clean)
    mean = sum(clean) / n
    med = clean[n // 2] if n % 2 == 1 else (clean[n // 2 - 1] + clean[n // 2]) / 2.0
    std = math.sqrt(sum((x - mean) ** 2 for x in clean) / n)
    return {
        "n": n,
        "min": round(clean[0], 4),
        "max": round(clean[-1], 4),
        "mean": round(mean, 4),
        "median": round(med, 4),
        "std": round(std, 4),
    }


def load_model_data(filename: str, cqe: CodeQualityEvaluator):
    path = PROCESSED / filename
    with open(path, "r", encoding="utf-8") as f:
        preds = json.load(f)

    times_all = []
    times_clean = []
    over_120 = 0
    over_360 = 0
    by_pair_time = {}
    by_pair_quality = {}
    by_frag = defaultdict(lambda: {"mi": [], "ccn": [], "nloc": []})

    for p in preds:
        key = (p["fragment_id"], p["task_id"])
        t = (p.get("llm_metadata") or {}).get("time_seconds")
        if t is not None:
            t = float(t)
            times_all.append(t)
            by_pair_time[key] = t
            if t <= TIMEOUT_THRESHOLD:
                times_clean.append(t)
            else:
                over_120 += 1
            if t > 360.0:
                over_360 += 1

        if p.get("compilation_status") == 1 and p.get("predicted_code"):
            q = cqe.evaluate(p["predicted_code"])
            if q and q.get("nloc", 0) > 0:
                by_pair_quality[key] = q
                by_frag[p["fragment_id"]]["mi"].append(q["mi"])
                by_frag[p["fragment_id"]]["ccn"].append(q["ccn"])
                by_frag[p["fragment_id"]]["nloc"].append(q["nloc"])

    frag_avg = {}
    for fid, d in by_frag.items():
        if d["mi"]:
            frag_avg[fid] = {
                "mi": sum(d["mi"]) / len(d["mi"]),
                "ccn": sum(d["ccn"]) / len(d["ccn"]),
                "nloc": sum(d["nloc"]) / len(d["nloc"]),
            }

    return {
        "times_all": times_all,
        "times_clean": times_clean,
        "over_120": over_120,
        "over_360": over_360,
        "by_pair_time": by_pair_time,
        "by_pair_quality": by_pair_quality,
        "frag_avg": frag_avg,
    }


def run_paired_test(a: list[float], b: list[float]) -> dict:
    w_stat, w_p = wilcoxon(a, b, alternative="two-sided")
    u_stat, u_p = mannwhitneyu(a, b, alternative="two-sided")
    diffs = [x - y for x, y in zip(a, b)]
    return {
        "n_pairs": len(a),
        "mean_a": round(sum(a) / len(a), 4),
        "median_a": safe_stats(a)["median"],
        "mean_b": round(sum(b) / len(b), 4),
        "median_b": safe_stats(b)["median"],
        "median_diff_a_minus_b": safe_stats(diffs)["median"],
        "wilcoxon_W": round(float(w_stat), 2),
        "wilcoxon_p": float(f"{w_p:.6e}"),
        "mannwhitney_U": round(float(u_stat), 2),
        "mannwhitney_p": float(f"{u_p:.6e}"),
    }


def main():
    cqe = CodeQualityEvaluator()
    print("Evaluating minimax predictions...")
    d_minimax = load_model_data("predictions_effective_java_paragraph_mbjp_minimax.json", cqe)
    print("Evaluating mimo_2_6 predictions...")
    d_mimo = load_model_data("predictions_effective_java_paragraph_mbjp_mimo_2_6.json", cqe)

    # Paired inference time (all & clean <= 120s)
    common_t_keys = sorted(set(d_mimo["by_pair_time"]) & set(d_minimax["by_pair_time"]))
    t_all_mimo = [d_mimo["by_pair_time"][k] for k in common_t_keys]
    t_all_mm = [d_minimax["by_pair_time"][k] for k in common_t_keys]

    clean_t_keys = [
        k for k in common_t_keys
        if d_mimo["by_pair_time"][k] <= TIMEOUT_THRESHOLD
        and d_minimax["by_pair_time"][k] <= TIMEOUT_THRESHOLD
    ]
    t_cln_mimo = [d_mimo["by_pair_time"][k] for k in clean_t_keys]
    t_cln_mm = [d_minimax["by_pair_time"][k] for k in clean_t_keys]

    # Paired fragment-level quality (n=410)
    common_fids = sorted(set(d_mimo["frag_avg"]) & set(d_minimax["frag_avg"]))
    frag_tests = {}
    for metric in ("mi", "ccn", "nloc"):
        va = [d_mimo["frag_avg"][f][metric] for f in common_fids]
        vb = [d_minimax["frag_avg"][f][metric] for f in common_fids]
        frag_tests[metric] = run_paired_test(va, vb)

    # Paired prediction-level quality (both compiled)
    common_q_keys = sorted(set(d_mimo["by_pair_quality"]) & set(d_minimax["by_pair_quality"]))
    pred_tests = {}
    for metric in ("mi", "ccn", "nloc"):
        va = [d_mimo["by_pair_quality"][k][metric] for k in common_q_keys]
        vb = [d_minimax["by_pair_quality"][k][metric] for k in common_q_keys]
        pred_tests[metric] = run_paired_test(va, vb)

    report = {
        "inference_time_summary": {
            "minimax": {
                "raw": safe_stats(d_minimax["times_all"]),
                "clean_le_120s": safe_stats(d_minimax["times_clean"]),
                "over_120s_count": d_minimax["over_120"],
                "over_120s_pct": round(d_minimax["over_120"] / len(d_minimax["times_all"]) * 100, 2),
                "over_360s_count": d_minimax["over_360"],
            },
            "mimo_2_6": {
                "raw": safe_stats(d_mimo["times_all"]),
                "clean_le_120s": safe_stats(d_mimo["times_clean"]),
                "over_120s_count": d_mimo["over_120"],
                "over_120s_pct": round(d_mimo["over_120"] / len(d_mimo["times_all"]) * 100, 2),
                "over_360s_count": d_mimo["over_360"],
            },
        },
        "wilcoxon_mimo_vs_minimax": {
            "time_all_pairs": run_paired_test(t_all_mimo, t_all_mm),
            "time_clean_le_120s_pairs": run_paired_test(t_cln_mimo, t_cln_mm),
            "quality_fragment_level_410": frag_tests,
            "quality_prediction_level_both_compiled": pred_tests,
        },
    }

    out_path = ROOT / "statistics" / "extra_statistical_tests.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 1. Redraw clean (<= 120s) inference time violin plot
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    vecs = [d_mimo["times_clean"], d_minimax["times_clean"]]
    labels = [
        f"MiMo-2.6-Flash\n(n={len(d_mimo['times_clean'])}, excl. >120s: {d_mimo['over_120']})",
        f"MiniMax-M2.5\n(n={len(d_minimax['times_clean'])}, excl. >120s: {d_minimax['over_120']})",
    ]
    parts = ax.violinplot(vecs, showmeans=True, showmedians=True, showextrema=True)
    for pc, col in zip(parts["bodies"], ["#4C72B0", "#55A868"]):
        pc.set_facecolor(col)
        pc.set_edgecolor("black")
        pc.set_alpha(0.65)
    parts["cmeans"].set_color("#C44E52")
    parts["cmedians"].set_color("black")
    ax.set_title("Stage 03 Single-Request Inference Time Distribution (≤ 120 s)")
    ax.set_ylabel("Inference Time (seconds)")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for idx_m, v in enumerate(vecs, 1):
        st = safe_stats(v)
        ax.annotate(
            f"median={st['median']:.2f}s\nmean={st['mean']:.2f}s",
            xy=(idx_m, st["median"]),
            xytext=(16, 12),
            textcoords="offset points",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.9),
        )
    plt.tight_layout()
    plt.savefig(PLOTS / "fig_stage03_inference_time_violin.png")
    plt.close(fig)

    # 2. New violin plots for MI, CCN, NLOC
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    specs = [
        ("mi", "Maintainability Index (MI)", "Average MI"),
        ("ccn", "Cyclomatic Complexity (CCN)", "Average CCN"),
        ("nloc", "Lines of Code (NLOC)", "Average NLOC"),
    ]
    for ax, (mk, title, ylabel) in zip(axes, specs):
        va = [d_mimo["frag_avg"][f][mk] for f in common_fids]
        vb = [d_minimax["frag_avg"][f][mk] for f in common_fids]
        parts = ax.violinplot([va, vb], showmeans=True, showmedians=True, showextrema=True)
        for pc, col in zip(parts["bodies"], ["#4C72B0", "#55A868"]):
            pc.set_facecolor(col)
            pc.set_edgecolor("black")
            pc.set_alpha(0.65)
        parts["cmeans"].set_color("#C44E52")
        parts["cmedians"].set_color("black")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["MiMo-2.6-Flash", "MiniMax-M2.5"])
        ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS / "fig_stage03_quality_violins.png")
    plt.close(fig)

    print("\n=== EXTRA STATS REPORT ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved JSON -> {out_path}")
    print(f"Saved plots -> {PLOTS / 'fig_stage03_inference_time_violin.png'} & fig_stage03_quality_violins.png")


if __name__ == "__main__":
    main()