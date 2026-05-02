"""
Evaluation script for comparing image similarity search methods.

Computes P@5, mAP@5, and Top-1 accuracy by matching retrieved image genres against
the query genre, and measures per-query latency. Supports two modes:
  - single: evaluates Baseline, Hybrid, and Reranker sequentially per query.
  - batch:  measures latency improvement of batched vs. single-query retrieval on baseline.

Results are printed as Rich tables to the console.
"""

import time
import pandas as pd
import numpy as np
from pathlib import Path

import config
from search.baseline import BaselineSearcher
from search.hybrid import HybridSearcher
from search.reranker import RerankSearcher

from typing import Union
from rich.console import Console
from rich.table import Table

console = Console()


# Metrics
def precison_at_k(retrieved_style, query_style, k=5):

    relevant = sum(1 for style in retrieved_style[:k] if style == query_style)

    return relevant / k


def average_precision_at_k(retrieved_style, query_style, k=5):
    relevant_count = 0
    precison = []

    for i, style in enumerate(retrieved_style[:k]):
        if style == query_style:
            relevant_count += 1
            precison.append(relevant_count / (i + 1))

    if not precison:
        return 0.0

    return sum(precison) / len(precison)


def accuracy_at_1(retrieved_style, query_style):
    return 1 if retrieved_style[0] == query_style else 0


# helper
def get_retrieved_styles(retrieved_ids, index_df):
    retrieved_style = []

    for rid in retrieved_ids:
        row = index_df[index_df["id"] == rid]

        if not row.empty:
            style = row["genre"].iloc[0].strip()
            retrieved_style.append(style)

    return retrieved_style


# main functions to evaluate each searcher
# sequential search of each query
def eval_searcher_single(
    searcher: Union[BaselineSearcher, RerankSearcher, HybridSearcher],
    query_df: pd.DataFrame,
    index_df: pd.DataFrame,
):

    p5 = []
    ap5 = []
    top1 = []
    latency = []

    for _, row in query_df.iterrows():
        image_path = config.WIKIART_DIR / row["filename"]

        with open(image_path, "rb") as f:
            image_bytes = f.read()
            start_time = time.perf_counter()
            ids = searcher.search(image_bytes=image_bytes)
            end_time = time.perf_counter()

        retrieved_style = get_retrieved_styles(ids, index_df)

        precison_at_k_score = precison_at_k(retrieved_style, row["genre"].strip())
        avg_precision_at_k_score = average_precision_at_k(
            retrieved_style, row["genre"].strip()
        )
        acc_at_1_score = accuracy_at_1(retrieved_style, row["genre"].strip())

        p5.append(precison_at_k_score)
        ap5.append(avg_precision_at_k_score)
        top1.append(acc_at_1_score)
        latency.append((end_time - start_time) * 1000)

    return {
        "precision_at_5": np.mean(p5),
        "average_precision_at_5": np.mean(ap5),
        "accuracy_at_1": np.mean(top1),
        "latency": np.mean(latency),
    }


def eval_searcher_batch(
    searcher,
    query_df,
):
    all_bytes = []
    for _, row in query_df.iterrows():
        image_path = config.WIKIART_DIR / row["filename"]
        all_bytes.append(open(image_path, "rb").read())

    start = time.perf_counter()
    all_ids = searcher.search_batch(all_bytes)
    end = time.perf_counter()

    elapsed = (end - start) * 1000

    return elapsed / len(all_bytes)


def summarize(metrics: dict) -> None:
    table = Table(title="Comparison Table")

    table.add_column(header="Method")
    table.add_column(header="P@5")
    table.add_column(header="mAP@5")
    table.add_column(header="Top-1")
    table.add_column(header="Latency (ms)")

    for method, metric in metrics.items():
        table.add_row(
            method,
            f"{metric["precision_at_5"]:.2f}",
            f"{metric["average_precision_at_5"]:.2f}",
            f"{metric["accuracy_at_1"]:.2f}",
            f"{metric["latency"]:.2f}",
        )

    console.print(table)


def run_eval(mode: str = "single"):
    # load csv files
    console.print("Reading CSV Files", style="magenta")
    query_df = pd.read_csv(config.QUERY_FILE)
    index_df = pd.read_csv(config.INDEX_FILE)

    results = {}

    if mode == "single":
        # baseline
        console.print("Evaluating Baseline Model", style="magenta")
        searcher = BaselineSearcher()
        results["Baseline"] = eval_searcher_single(searcher, query_df, index_df)
        searcher.close()

        # hybrid
        console.print("Evaluating Hybrid Model", style="magenta")
        searcher = HybridSearcher()
        results["Hybrid"] = eval_searcher_single(searcher, query_df, index_df)
        searcher.close()

        # reranker
        console.print("Evaluating Reranker Model", style="magenta")
        baseline = BaselineSearcher()
        searcher = RerankSearcher(baseline=baseline)
        results["Re-ranked"] = eval_searcher_single(searcher, query_df, index_df)
        baseline.close()

        console.print("Evaluation Complete, Printing Summary...", style="magenta")
        summarize(metrics=results)

    elif mode == "batch":
        # checking only latency improvement
        searcher = BaselineSearcher()
        latency_batch = eval_searcher_batch(searcher, query_df)
        latency_single = eval_searcher_single(searcher, query_df, index_df)["latency"]
        searcher.close()

        table = Table(title="Latency Optimization (Batched Retrieval)")
        table.add_column(header="Method")
        table.add_column(header="Before")
        table.add_column(header="After")

        table.add_row("Baseline", f"{latency_single:.2f}", f"{latency_batch:.2f}")

        console.print(table)


if __name__ == "__main__":
    import sys

    mode = sys.argv[1]
    run_eval(mode=mode)
