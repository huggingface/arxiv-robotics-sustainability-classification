#!/usr/bin/env python3
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

VALID_IMPACT_TYPES = {"all", "any", "social", "ecological", "sustainability", "un_sdgs"}


def _get_parsed(paper: dict) -> dict:
    """Return pre-parsed classification fields from a paper record.
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
                    paper_id = paper.get("id")
                    if paper_id and paper_id in seen_ids:
                        continue
                    if paper_id:
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
    max_unit = 12 if granularity == "month" else 4

    while (a < end_a) or (a == end_a and b <= end_b):
        out.append((a, b))
        b += 1
        if b > max_unit:
            b = 1
            a += 1
    return out


def format_label(period_key, granularity: str):
    year, unit = period_key
    if granularity == "month":
        return f"{year}-{unit:02d}"
    return f"{year} Q{unit}"



def impact_label(impact_type: str):
    labels = {
        "all": "All Impact Mentions",
        "any": "Any Impact Mention",
        "social": "Social Impact Mention",
        "ecological": "Ecological Impact Mention",
        "sustainability": "Sustainability Impact Mention",
        "un_sdgs": "UN SDGs Mention",
    }
    return labels[impact_type]


def compute_percentage(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    pct = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0,
    ) * 100
    pct[denominator <= 0] = 0.0
    return pct


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {os.path.basename(sys.argv[0])} <papers.json> [quarter|month] [all|any|social|ecological|sustainability|un_sdgs]")
        sys.exit(1)

    filename = sys.argv[1]
    granularity = sys.argv[2].lower() if len(sys.argv) > 2 else "quarter"
    impact_type = sys.argv[3].lower() if len(sys.argv) > 3 else "any"

    if granularity not in {"quarter", "month"}:
        print("Granularity must be 'quarter' or 'month'.")
        sys.exit(1)

    if impact_type not in VALID_IMPACT_TYPES:
        print("Impact type must be one of: all, any, social, ecological, sustainability, un_sdgs")
        sys.exit(1)

    papers = load_papers(filename)

    total_per_period = defaultdict(int)
    impact_per_period_by_type = {k: defaultdict(int) for k in ("any", "social", "ecological", "sustainability", "un_sdgs")}

    for paper in papers:
        period_key = parse_period(paper.get("published"), granularity)
        if not period_key:
            continue

        total_per_period[period_key] += 1

        p = _get_parsed(paper)
        flags = {
            "social": bool(p.get("mentions_social_impact")),
            "ecological": bool(p.get("mentions_ecological_impact")),
            "sustainability": bool(p.get("mentions_sustainability_impact")),
            "un_sdgs": bool(p.get("mentions_un_sdgs")),
        }
        flags["any"] = any(flags[k] for k in ("social", "ecological", "sustainability", "un_sdgs"))
        for k in impact_per_period_by_type:
            if flags[k]:
                impact_per_period_by_type[k][period_key] += 1

    period_keys = build_period_range(set(total_per_period.keys()), granularity)
    if not period_keys:
        print("No valid date data found.")
        sys.exit(1)

    labels = [format_label(k, granularity) for k in period_keys]
    x = np.arange(len(period_keys))

    total_vals = np.array([total_per_period.get(k, 0) for k in period_keys], dtype=float)
    if impact_type == "all":
        fig, ax1 = plt.subplots(figsize=(12, 7))
        bars_total = ax1.bar(x, total_vals, width=0.82, color="lightgray", label="Total Papers", zorder=1)
        ax1.set_ylabel("Number of Papers")
        ax1.set_xlabel("Year and Quarter" if granularity == "quarter" else "Year and Month")

        ax2 = ax1.twinx()
        colors = {
            "social": "#ff7f0e",
            "ecological": "#2ca02c",
            "sustainability": "#d62728",
            "un_sdgs": "#9467bd",
        }

        lines = []
        for k in ("social", "ecological", "sustainability", "un_sdgs"):
            vals = np.array([impact_per_period_by_type[k].get(pk, 0) for pk in period_keys], dtype=float)
            pct = compute_percentage(vals, total_vals)
            line, = ax2.plot(
                x,
                pct,
                marker="o",
                linewidth=1.8,
                color=colors[k],
                label=f"{impact_label(k)} / Total (%)",
                zorder=3,
            )
            lines.append(line)

            total_all = int(total_vals.sum())
            impact_all = int(vals.sum())
            pct_all = (impact_all / total_all * 100) if total_all else 0.0
            print(f"Overall ({k}): {impact_all} / {total_all} ({pct_all:.2f}%)")

        ax2.set_ylabel("Impact Mentioned Papers (%)")
        ax1.set_title("All Impact Types (excluding Any) vs Total Papers")

        handles = [bars_total] + lines
        ax1.legend(handles, [h.get_label() for h in handles], loc="upper left", frameon=True)
    else:
        impact_vals = np.array([impact_per_period_by_type[impact_type].get(k, 0) for k in period_keys], dtype=float)
        impact_pct = compute_percentage(impact_vals, total_vals)

        fig, ax1 = plt.subplots(figsize=(12, 7))

        bars_total = ax1.bar(x, total_vals, width=0.82, color="lightgray", label="Total Papers", zorder=1)
        bars_impact = ax1.bar(x, impact_vals, width=0.52, color="#2ca02c", label=impact_label(impact_type), zorder=2)

        ax1.set_ylabel("Number of Papers")
        ax1.set_xlabel("Year and Quarter" if granularity == "quarter" else "Year and Month")

        ax2 = ax1.twinx()
        line_pct, = ax2.plot(x, impact_pct, color="#1f77b4", marker="o", linewidth=1.8,
                             label=f"{impact_label(impact_type)} / Total (%)", zorder=3)
        ax2.set_ylabel("Impact Mentioned Papers (%)")

        ax1.set_title(f"{impact_label(impact_type)} vs Total Papers")

        handles = [bars_total, bars_impact, line_pct]
        ax1.legend(handles, [h.get_label() for h in handles], loc="upper left", frameon=True)

        total_all = int(total_vals.sum())
        impact_all = int(impact_vals.sum())
        pct_all = (impact_all / total_all * 100) if total_all else 0.0
        print(f"Overall ({impact_type}): {impact_all} / {total_all} ({pct_all:.2f}%)")

    if granularity == "month":
        step = 3 if len(labels) > 18 else 1
    else:
        step = 2 if len(labels) > 14 else 1

    tick_idx = list(range(0, len(labels), step))
    ax1.set_xticks(x[tick_idx])
    ax1.set_xticklabels([labels[i] for i in tick_idx], rotation=45, ha="right")

    ax1.grid(False)
    ax2.grid(False)
    ax1.set_xlim(-0.7, len(x) - 0.3)
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
