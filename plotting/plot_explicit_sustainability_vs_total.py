#!/usr/bin/env python3
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


# Styling aligned with plot_total_in_time.py
TOTAL_COLOR = "#E0E0E0"
EXPLICIT_COLOR = "#59a14f"
PCT_LINE_COLOR = "#4e79a7"


def _get_parsed(paper: dict) -> dict:
    """Return pre-parsed classification fields.
    Supports both compiled (top-level) and raw results.json (deepseek.parsed) formats.
    """
    if "motivated_by_sdgs" in paper or "mentions_social_impact" in paper:
        return paper
    return (paper.get("deepseek") or {}).get("parsed") or {}


def load_papers(filename: str):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            papers = json.load(f)
        print(f"Loaded {len(papers)} papers from {filename}")
        return papers

    papers = []
    seen_ids = set()
    found = False
    basename = os.path.basename(filename)

    for root, _, files in os.walk("."):
        if basename in files:
            found = True
            filepath = os.path.join(root, basename)
            print(f"Found {basename} in {root}")
            with open(filepath, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                for paper in loaded:
                    #if "API Error" in paper.get("sdg_analysis", ""):
                    #    continue
                    paper_id = paper.get("id")
                    if paper_id:
                        if paper_id in seen_ids:
                            continue
                        seen_ids.add(paper_id)
                    papers.append(paper)

    if not found:
        print(f"File {filename} not found in current directory or subdirectories.")
        sys.exit(1)

    print(f"Loaded {len(papers)} papers from matches of {basename}")
    return papers


def parse_period(published: str, granularity: str):
    if not published:
        return None
    try:
        dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
    except Exception:
        return None

    if granularity == "month":
        return dt.year, dt.month

    quarter = (dt.month - 1) // 3 + 1
    return dt.year, quarter


def build_period_range(keys, granularity: str):
    if not keys:
        return []

    start_a, start_b = min(keys)
    end_a, end_b = max(keys)

    out = []
    a, b = start_a, start_b
    end_step = 12 if granularity == "month" else 4

    while (a < end_a) or (a == end_a and b <= end_b):
        out.append((a, b))
        b += 1
        if b > end_step:
            b = 1
            a += 1
    return out


def format_label(period_key, granularity: str):
    year, unit = period_key
    if granularity == "month":
        return f"{year}-{unit:02d}"
    return f"{year} Q{unit}"


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {os.path.basename(sys.argv[0])} <papers.json> [quarter|month]")
        sys.exit(1)

    filename = sys.argv[1]
    granularity = sys.argv[2].lower() if len(sys.argv) > 2 else "quarter"

    if granularity not in {"quarter", "month"}:
        print("Granularity must be 'quarter' or 'month'.")
        sys.exit(1)

    papers = load_papers(filename)

    total_per_period = defaultdict(int)
    explicit_per_period = defaultdict(int)

    for paper in papers:
        period_key = parse_period(paper.get("published"), granularity)
        if not period_key:
            continue

        total_per_period[period_key] += 1

        p = _get_parsed(paper)
        if p.get("motivated_by_sdgs"):
            explicit_per_period[period_key] += 1

    period_keys = build_period_range(set(total_per_period.keys()), granularity)
    if not period_keys:
        print("No valid date data found.")
        sys.exit(1)

    labels = [format_label(k, granularity) for k in period_keys]
    x = np.arange(len(period_keys))

    total_vals = np.array([total_per_period.get(k, 0) for k in period_keys], dtype=float)
    explicit_vals = np.array([explicit_per_period.get(k, 0) for k in period_keys], dtype=float)
    explicit_pct = np.divide(explicit_vals, total_vals, out=np.zeros_like(explicit_vals), where=total_vals > 0) * 100
    fig, ax1 = plt.subplots(figsize=(16, 6), facecolor="#FFFFFF")
    ax1.set_facecolor("#FFFFFF")

    bars_total = ax1.bar(x, total_vals, width=0.82, color=TOTAL_COLOR, label="Total Papers", zorder=1)
    bars_explicit = ax1.bar(x, explicit_vals, width=0.82, color=EXPLICIT_COLOR, label="Explicit Sustainability Motivation", zorder=2)

    ax1.set_ylabel("Number of Papers", fontsize=17, color="#333333", labelpad=10)
    ax1.set_xlabel("Year and Quarter" if granularity == "quarter" else "Year and Month", fontsize=17, color="#333333", labelpad=10)

    ax2 = ax1.twinx()
    line_pct, = ax2.plot(x, explicit_pct, color=PCT_LINE_COLOR, marker="o", linewidth=1.8,
                         label="Explicit / Total (%)", zorder=3)
    ax2.set_ylabel("Explicitly Motivated Papers (%)", fontsize=17, color="#333333", labelpad=10)

    if granularity == "month":
        step = 3 if len(labels) > 18 else 1
    else:
        step = 2 if len(labels) > 14 else 1
    tick_idx = list(range(0, len(labels), step))
    ax1.set_xticks(x[tick_idx])
    ax1.set_xticklabels([labels[i] for i in tick_idx], rotation=45, ha="right", fontsize=13, color="#111111")
    ax1.tick_params(axis="y", labelsize=13, color="#111111")
    ax2.tick_params(axis="y", labelsize=13, color="#111111")

    handles = [bars_total, bars_explicit, line_pct]
    ax1.legend(
        handles,
        [h.get_label() for h in handles],
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=15,
        handlelength=1.2,
        handleheight=1.2,
    )

    ax1.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.3, color="#B0B0B0")
    ax1.set_axisbelow(True)
    ax2.grid(False)
    ax1.set_xlim(-0.7, len(x) - 0.3)

    # Clean borders to match the reference style.
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.spines["bottom"].set_color("#333333")
    ax1.tick_params(axis="y", which="both", left=False)

    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_color("#333333")

    total_all = int(total_vals.sum())
    explicit_all = int(explicit_vals.sum())
    pct_all = (explicit_all / total_all * 100) if total_all else 0.0
    print(f"Overall: explicit={explicit_all} / total={total_all} ({pct_all:.2f}%)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
