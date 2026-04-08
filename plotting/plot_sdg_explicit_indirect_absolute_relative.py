#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np

# SDG_NAMES = [
#     "No Poverty",
#     "Zero Hunger",
#     "Good Health & Well-being",
#     "Quality Education",
#     "Gender Equality",
#     "Clean Water & Sanitation",
#     "Affordable & Clean Energy",
#     "Decent Work & Economic Growth",
#     "Industry, Innovation & Infrastructure",
#     "Reduced Inequalities",
#     "Sustainable Cities & Communities",
#     "Responsible Consumption & Production",
#     "Climate Action",
#     "Life Below Water",
#     "Life On Land",
#     "Peace, Justice & Strong Institutions",
#     "Partnerships for the Goals",
# ]

SDG_NAMES = [   "No Poverty",
    "No Hunger",
    "Health",
    "Education",
    "Gender Eq.",
    "Clean Water",
    "Clean Energy",
    "Decent Work",
    "Industry",
    "Reduce Ineq.",
    "Sust. Cities",
    "Consumption",
    "Climate Action",
    "Life In Water",
    "Life On Land",
    "Peace & Justice",
    "Partnerships"
]

# Colors aligned with the HTML embeds.
COLOR_EXPLICIT = "#59a14f"  # SDG Motivated
COLOR_ALIGNED = "#9fa8b3"   # SDG Aligned
COLOR_NONE = "#e15759"      # No SDG Relevance


def paper_dedupe_key(paper: dict) -> str:
    paper_id = paper.get("id")
    if paper_id:
        return str(paper_id).strip()
    title = paper.get("title")
    published = paper.get("published")
    if title and published:
        return f"{title.strip()}::{published}"
    if title:
        return f"title::{title.strip()}"
    return ""


def load_json_file(path: str) -> List[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception as exc:
        print(f"Error loading {path}: {exc}")
    return []


def load_papers(input_file: str, dataset_dir: str) -> List[dict]:
    papers: List[dict] = []
    seen: Set[str] = set()

    def append_many(items: List[dict], source_label: str) -> None:
        added = 0
        for paper in items:
            key = paper_dedupe_key(paper)
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            papers.append(paper)
            added += 1
        if added:
            print(f"  Loaded {added} papers from {source_label}")

    if input_file and os.path.exists(input_file):
        print(f"Loading specific file: {input_file}")
        append_many(load_json_file(input_file), input_file)
        print(f"Total loaded papers: {len(papers)}")
        return papers

    root_dir = dataset_dir if os.path.exists(dataset_dir) else "."
    print(f"Scanning {root_dir} for annotated metadata JSONs...")

    candidates: List[str] = []
    for root, _, files in os.walk(root_dir):
        for name in files:
            if name in {
                "metadata_for_downloaded_pdfs_sdg_annotated_full_text_ifr.json",
                "metadata_for_downloaded_pdfs_sdg_annotated_full_text.json",
            }:
                candidates.append(os.path.join(root, name))

    for path in sorted(candidates):
        rel = os.path.relpath(path, root_dir)
        append_many(load_json_file(path), rel)

    print(f"Total loaded papers: {len(papers)}")
    return papers


def _get_parsed(paper: dict) -> dict:
    """Return pre-parsed classification fields from a paper record.
    Supports both compiled (top-level) and raw results.json (deepseek.parsed) formats.
    """
    if "motivated_by_sdgs" in paper or "mentions_social_impact" in paper:
        return paper
    return (paper.get("deepseek") or {}).get("parsed") or {}


def _sdg_nums(sdg_list) -> Set[int]:
    nums: Set[int] = set()
    for s in (sdg_list or []):
        m = re.search(r"\d+", str(s))
        if m:
            nums.add(int(m.group(0)))
    return nums


def build_counts(papers: List[dict]) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, int]:
    n_papers = len(papers)
    if n_papers == 0:
        raise ValueError("No papers found.")

    explicit_counts = {i: 0 for i in range(1, 18)}
    indirect_counts = {i: 0 for i in range(1, 18)}
    no_sdgs_count = 0

    for paper in papers:
        p = _get_parsed(paper)
        explicit_sdgs = _sdg_nums(p.get("motivated_by_sdgs"))
        indirect_sdgs = _sdg_nums(p.get("aligned_with_sdgs"))

        if not explicit_sdgs and not indirect_sdgs:
            no_sdgs_count += 1
            continue

        for sdg_id in explicit_sdgs:
            explicit_counts[sdg_id] += 1
            indirect_counts[sdg_id] += 1  # Count explicit SDGs as also aligned.
        for sdg_id in indirect_sdgs:
            if sdg_id not in explicit_sdgs:
                indirect_counts[sdg_id] += 1

    # Match the HTML ordering: ascending by aligned/indirect count.
    sorted_ids = sorted(range(1, 18), key=lambda sdg_id: indirect_counts[sdg_id])

    labels = ["No SDGs"] + [f"SDG {sdg_id}: {SDG_NAMES[sdg_id - 1]}" for sdg_id in sorted_ids]
    explicit_series = np.array([0] + [explicit_counts[sdg_id] for sdg_id in sorted_ids], dtype=float)
    indirect_series = np.array([0] + [indirect_counts[sdg_id] for sdg_id in sorted_ids], dtype=float)
    none_series = np.array([no_sdgs_count] + [0] * len(sorted_ids), dtype=float)

    return labels, explicit_series, indirect_series, none_series, n_papers


def plot_absolute(labels: List[str], explicit: np.ndarray, aligned: np.ndarray, none: np.ndarray, n_papers: int):
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(12, 9), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    ax.barh(y, aligned, height=0.72, color=COLOR_ALIGNED, alpha=0.45, label="SDG Aligned", zorder=1)
    ax.barh(y, explicit, height=0.72, color=COLOR_EXPLICIT, label="SDG Motivated", zorder=2)
    ax.barh(y, none, height=0.72, color=COLOR_NONE, label="No SDG Relevance", zorder=3)

    # Label aligned and no-sdg percentages as share of total papers (matches embed intent).
    for i, (label, a_val) in enumerate(zip(labels, aligned)):
        if label == "No SDGs" or n_papers <= 0 or a_val <= 0:
            continue
        pct = (a_val / n_papers) * 100
        ax.text(a_val + 300, i, f"{pct:.1f}%", va="center", ha="left", fontsize=11, color="#333333")

    for i, n_val in enumerate(none):
        if n_papers <= 0 or n_val <= 0:
            continue
        pct = (n_val / n_papers) * 100
        ax.text(n_val + 300, i, f"{pct:.1f}%", va="center", ha="left", fontsize=11, color=COLOR_NONE)

    max_value = float(np.max(np.vstack([explicit, aligned, none]))) if len(labels) else 0.0
    ax.set_xlim(0, max(1.0, max_value * 1.3))

    ax.set_xlabel("Number of Papers (Count)", fontsize=13, color="#333333", labelpad=10)
    ax.set_ylabel("SDG Category", fontsize=13, color="#333333", labelpad=10)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12, color="#111111")
    ax.tick_params(axis="x", labelsize=13, color="#111111")

    ax.grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.3, color="#B0B0B0")
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", which="both", left=False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


def plot_relative(labels: List[str], explicit: np.ndarray, aligned: np.ndarray):
    # Exclude "No SDGs" row for relative SDG-only view.
    labels_sdg = labels[1:]
    explicit_sdg = explicit[1:]
    aligned_sdg = aligned[1:]

    aligned_not_motivated = np.maximum(aligned_sdg - explicit_sdg, 0)

    explicit_rel = np.divide(
        explicit_sdg,
        aligned_sdg,
        out=np.zeros_like(explicit_sdg, dtype=float),
        where=aligned_sdg > 0,
    ) * 100

    aligned_rel = np.divide(
        aligned_not_motivated,
        aligned_sdg,
        out=np.zeros_like(aligned_not_motivated, dtype=float),
        where=aligned_sdg > 0,
    ) * 100

    # Match embed behavior: sort ascending by explicit relative share.
    sorted_idx = np.argsort(explicit_rel)
    labels_sorted = [labels_sdg[i] for i in sorted_idx]
    explicit_rel_sorted = explicit_rel[sorted_idx]
    aligned_rel_sorted = aligned_rel[sorted_idx]

    y = np.arange(len(labels_sorted))

    fig, ax = plt.subplots(figsize=(12, 9), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    ax.barh(y, explicit_rel_sorted, height=0.72, color=COLOR_EXPLICIT, label="SDG Motivated", zorder=2)
    # ax.barh(
    #     y,
    #     aligned_rel_sorted,
    #     left=explicit_rel_sorted,
    #     height=0.60,
    #     color=COLOR_ALIGNED,
    #     alpha=0.75,
    #     label="SDG Aligned (But not motivated by it!)",
    #     zorder=1,
    # )

    for i, val in enumerate(explicit_rel_sorted):
        if val > 50:
            ax.text(val / 2.0, i, f"{val:.1f}%", va="center", ha="center", fontsize=10, color="#FFFFFF")
        else:
            # For smaller percentages, place the label outside the bar for better visibility.
            ax.text(val + 1, i, f"{val:.1f}%", va="center", ha="left", fontsize=10, color="#333333")
    for i, val in enumerate(aligned_rel_sorted):
        if val > 0:
            x_pos = explicit_rel_sorted[i] + val / 2.0
            ax.text(x_pos, i, f"{val:.1f}%", va="center", ha="center", fontsize=10, color="#FFFFFF")

    ax.set_xlim(0, 100)
    ax.set_xlabel("Share Within SDG (%)", fontsize=13, color="#333333", labelpad=10)
    ax.set_ylabel("SDG Category", fontsize=13, color="#333333", labelpad=10)

    ax.set_yticks(y)
    ax.set_yticklabels(labels_sorted, fontsize=12, color="#111111")
    ax.tick_params(axis="x", labelsize=13, color="#111111")

    ax.grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.3, color="#B0B0B0")
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", which="both", left=False)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SDG explicit/aligned absolute and relative plots (Matplotlib)."
    )
    parser.add_argument(
        "papers_json",
        nargs="?",
        default="",
        help="Optional path to a specific papers JSON file. If omitted, scans annotated metadata files.",
    )
    parser.add_argument(
        "-i",
        "--input-file",
        default="",
        help="Path to metadata input JSON file (overrides positional papers_json).",
    )
    parser.add_argument(
        "--dataset-dir",
        default=".",
        help="Dataset root to scan when input file is not provided.",
    )
    args = parser.parse_args()

    input_file = (args.input_file or args.papers_json or "").strip()
    papers = load_papers(input_file, args.dataset_dir)
    if not papers:
        print("No papers found.")
        sys.exit(1)

    labels, explicit, aligned, none, n_papers = build_counts(papers)

    fig_abs = plot_absolute(labels, explicit, aligned, none, n_papers)
    fig_rel = plot_relative(labels, explicit, aligned)

    plt.show()


if __name__ == "__main__":
    main()
