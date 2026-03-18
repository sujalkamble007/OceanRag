"""
results_exporter.py — CSV export, leaderboard, comparison tables, chart data.
"""

import json
from pathlib import Path
from statistics import mean

import pandas as pd


# ─── Function 1: Export to CSV ──────────────────────────────────────────────

def export_to_csv(results: list) -> str:
    """
    Export all experiment results to CSV files.
    Creates main matrix CSV plus split metric CSVs for Phase 5.
    """
    Path("output").mkdir(exist_ok=True)

    df = pd.DataFrame(results)

    # Main export
    main_path = "output/experiment_matrix.csv"
    df.to_csv(main_path, index=False)

    # Retrieval metrics split
    retrieval_cols = [
        "chunk_strategy", "embedding_model", "retriever_type", "llm_name",
        "top_k", "precision_at_k", "recall_at_k", "mrr", "hit_rate",
    ]
    available_ret = [c for c in retrieval_cols if c in df.columns]
    df[available_ret].to_csv("output/retrieval_metrics.csv", index=False)

    # Generation metrics split
    gen_cols = [
        "chunk_strategy", "embedding_model", "retriever_type", "llm_name",
        "faithfulness", "answer_relevancy", "rouge_l", "latency_seconds", "cost_per_query",
    ]
    available_gen = [c for c in gen_cols if c in df.columns]
    df[available_gen].to_csv("output/generation_metrics.csv", index=False)

    print(f"📊 Exported → {main_path} ({len(df)} rows)")
    print(f"📊 Exported → output/retrieval_metrics.csv")
    print(f"📊 Exported → output/generation_metrics.csv")
    return main_path


# ─── Function 2: Print Leaderboard ─────────────────────────────────────────

def print_leaderboard(results: list, top_n: int = 10):
    """Print ranked table of experiment results by composite score."""
    if not results:
        print("  No results to display.")
        return

    # Compute composite score for each result
    scored = []
    for r in results:
        composite = (
            (r.get("precision_at_k") or 0) +
            (r.get("recall_at_k") or 0) +
            (r.get("faithfulness") or 0) +
            (r.get("answer_relevancy") or 0)
        ) / 4
        scored.append({**r, "composite": round(composite, 4)})

    scored.sort(key=lambda x: x["composite"], reverse=True)
    top = scored[:top_n]

    print()
    print("═" * 78)
    print(f"  DEEPRAG EXPERIMENT LEADERBOARD (Top {min(top_n, len(top))})")
    print("═" * 78)
    print(f"{'Rank':>4} | {'Chunk':<15} | {'Embed':<6} | {'Retriever':<10} | {'LLM':<20} | {'K':>2} | {'Score':>5}")
    print("-" * 78)

    for i, r in enumerate(top, 1):
        chunk = r.get("chunk_strategy", "?")[:15]
        embed = r.get("embedding_model", "?")[:6]
        retriever = r.get("retriever_type", "?")[:10]
        llm = r.get("llm_name", "?")[:20]
        k = r.get("top_k", "?")
        score = r.get("composite", 0)
        print(f"{i:>4} | {chunk:<15} | {embed:<6} | {retriever:<10} | {llm:<20} | {k:>2} | {score:.3f}")

    print("═" * 78)

    if scored:
        best = scored[0]
        print(f"🏆 Best: {best.get('chunk_strategy')} + {best.get('embedding_model')} + "
              f"{best.get('retriever_type')} + {best.get('llm_name')} + "
              f"k={best.get('top_k')} → Score: {best['composite']:.3f}")
    print()


# ─── Function 3: Print Metric Comparison Table ─────────────────────────────

def print_metric_comparison_table(results: list):
    """Print grouped metric comparisons by retriever, LLM, and chunk strategy."""
    if not results:
        return

    df = pd.DataFrame(results)

    # ── By Retriever ─────────────────────────────────────────────────
    print("\n── By Retriever ──────────────────────────────────────────────")
    print(f"{'Retriever':<12} | {'Precision':>9} | {'Recall':>7} | {'MRR':>5} | {'Hit Rate':>8} | {'Faith':>5} | {'Latency':>7}")
    print("-" * 70)

    if "retriever_type" in df.columns:
        for name, group in df.groupby("retriever_type"):
            p = group["precision_at_k"].mean() if "precision_at_k" in group else 0
            r = group["recall_at_k"].mean() if "recall_at_k" in group else 0
            m = group["mrr"].mean() if "mrr" in group else 0
            h = group["hit_rate"].mean() if "hit_rate" in group else 0
            f = group["faithfulness"].mean() if "faithfulness" in group else 0
            l = group["latency_seconds"].mean() if "latency_seconds" in group else 0
            print(f"{str(name):<12} | {p:>9.3f} | {r:>7.3f} | {m:>5.3f} | {h:>8.3f} | {f:>5.3f} | {l:>6.1f}s")

    # ── By LLM ───────────────────────────────────────────────────────
    print("\n── By LLM ────────────────────────────────────────────────────")
    print(f"{'LLM':<22} | {'Faith':>5} | {'Ans.Rel':>7} | {'ROUGE-L':>7} | {'Latency':>7} | {'Cost':>6}")
    print("-" * 68)

    if "llm_name" in df.columns:
        for name, group in df.groupby("llm_name"):
            f = group["faithfulness"].mean() if "faithfulness" in group else 0
            ar = group["answer_relevancy"].mean() if "answer_relevancy" in group else 0
            rl = group["rouge_l"].mean() if "rouge_l" in group else 0
            l = group["latency_seconds"].mean() if "latency_seconds" in group else 0
            c = group["cost_per_query"].mean() if "cost_per_query" in group else 0
            cost_str = "FREE" if c == 0 else f"${c:.4f}"
            print(f"{str(name):<22} | {f:>5.3f} | {ar:>7.3f} | {rl:>7.3f} | {l:>6.1f}s | {cost_str:>6}")

    # ── By Chunking ──────────────────────────────────────────────────
    print("\n── By Chunking ───────────────────────────────────────────────")
    print(f"{'Strategy':<16} | {'Precision':>9} | {'Recall':>7}")
    print("-" * 40)

    if "chunk_strategy" in df.columns:
        for name, group in df.groupby("chunk_strategy"):
            p = group["precision_at_k"].mean() if "precision_at_k" in group else 0
            r = group["recall_at_k"].mean() if "recall_at_k" in group else 0
            print(f"{str(name):<16} | {p:>9.3f} | {r:>7.3f}")

    print()


# ─── Function 4: Generate Chart Data ───────────────────────────────────────

def generate_chart_data(results: list) -> dict:
    """
    Produce pre-formatted JSON for Phase 5 React charts.
    Saves to output/chart_data.json.
    """
    if not results:
        return {}

    df = pd.DataFrame(results)

    # ── precision_recall_scatter ─────────────────────────────────────
    precision_recall_scatter = []
    if "retriever_type" in df.columns:
        for name, group in df.groupby("retriever_type"):
            data_points = []
            # We can plot each individual chunking strategy as a dot for this retriever
            for _, row in group.drop_duplicates(subset=['chunk_strategy']).iterrows():
                data_points.append({
                    "name": str(row.get("chunk_strategy", "Config")),
                    "precision": round(float(row["precision_at_k"]), 4),
                    "recall": round(float(row["recall_at_k"]), 4),
                })
            precision_recall_scatter.append({
                "retriever": str(name),
                "data": data_points
            })

    # ── latency_bar ──────────────────────────────────────────────────
    latency_bar = []
    if "llm_name" in df.columns:
        for name, group in df.groupby("llm_name"):
            provider = "Groq" if "Groq" in str(name) else "HuggingFace"
            latency_bar.append({
                "llm_name": str(name),
                "provider": provider,
                "avg_latency": round(float(group["latency_seconds"].mean()), 3),
            })
        latency_bar.sort(key=lambda x: x["avg_latency"])

    # ── retriever_radar ──────────────────────────────────────────────
    retriever_radar = []
    if "retriever_type" in df.columns:
        metrics = ["precision_at_k", "recall_at_k", "mrr", "hit_rate", "faithfulness"]
        metric_display = ["Precision", "Recall", "MRR", "Hit Rate", "Faith"]
        
        avgs = {}
        for name, group in df.groupby("retriever_type"):
            name_lower = str(name).lower()
            avgs[name_lower] = {m: round(float(group[m].mean()), 4) for m in metrics}
            
        for i, m in enumerate(metrics):
            radar_row = {"metric": metric_display[i]}
            for r_name in ["similarity", "mmr", "hybrid"]:
                radar_row[r_name] = avgs.get(r_name, {}).get(m, 0.0)
            retriever_radar.append(radar_row)

    # ── chunking_impact (Bar) ─────────────────────────────────────────
    chunking_impact = []
    if "chunk_strategy" in df.columns:
        for name, group in df.groupby("chunk_strategy"):
            chunking_impact.append({
                "strategy": str(name),
                "precision": round(float(group["precision_at_k"].mean()), 4),
                "recall": round(float(group["recall_at_k"].mean()), 4),
            })
        chunking_impact.sort(key=lambda x: x["recall"], reverse=True)

    # ── embedding_comparison (Bar) ───────────────────────────────────
    embedding_comparison = []
    if "embedding_model" in df.columns:
        for name, group in df.groupby("embedding_model"):
            embedding_comparison.append({
                "model": str(name),
                "precision": round(float(group["precision_at_k"].mean()), 4),
                "faithfulness": round(float(group["faithfulness"].mean()), 4),
            })
        embedding_comparison.sort(key=lambda x: x["precision"], reverse=True)

    # ── cost_vs_quality (Scatter) ────────────────────────────────────
    cost_vs_quality = []
    if "llm_name" in df.columns:
        for name, group in df.groupby("llm_name"):
            provider = "Groq" if "Groq" in str(name) else "HuggingFace"
            composite = (
                group["precision_at_k"].mean() +
                group["recall_at_k"].mean() +
                group["faithfulness"].mean() +
                group["answer_relevancy"].mean()
            ) / 4
            cost_vs_quality.append({
                "llm": str(name),
                "cost": round(float(group["cost_per_query"].mean()), 6),
                "quality": round(float(composite), 4),
                "provider": provider,
            })

    chart_data = {
        "precision_recall_scatter": precision_recall_scatter,
        "latency_bar": latency_bar,
        "retriever_radar": retriever_radar,
        "chunking_impact": chunking_impact,
        "embedding_comparison": embedding_comparison,
        "cost_vs_quality": cost_vs_quality,
    }

    Path("output").mkdir(exist_ok=True)
    with open("output/chart_data.json", "w") as f:
        json.dump(chart_data, f, indent=2, default=str)

    print("📊 Chart data saved → output/chart_data.json")
    return chart_data
