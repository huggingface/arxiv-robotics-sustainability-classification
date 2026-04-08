#!/usr/bin/env python3
import json
import os
import sys

import matplotlib.pyplot as plt

# Exact hex colors from your preferred palette
COLORS = {
    "no_impact": "#E3E3E3",       # Light Gray
    "impact_any": "#4e79a7",      # Steel Blue
    "social": "#D68B63",          # Muted Orange
    "ecological": "#B6BB66",      # Olive
    "sustainability": "#62C1A4",  # Teal
    "un_sdg": "#59a14f",          # Greenish
}


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


def safe_pct(value: int, total: int) -> float:
    return (100.0 * value / total) if total > 0 else 0.0


def draw_binary_donut(ax, positive_pct: float, positive_label: str, negative_label: str, positive_color: str, center_text: str, title: str):
    negative_pct = 100.0 - positive_pct
    angle = (positive_pct / 100.0) * 360.0
    start_angle = angle / 2.0

    ax.pie(
        [positive_pct, negative_pct],
        colors=[positive_color, COLORS["no_impact"]],
        startangle=start_angle,
        counterclock=False,
        wedgeprops=dict(width=0.6, edgecolor="white", linewidth=1.5),
    )

    ax.text(0, 0, center_text, ha="center", va="center", fontsize=17, color="#333333")
    ax.set_title(title, fontsize=15, color="#333333")

    if negative_pct > 0:
        ax.text(-0.7, 0, f"{negative_label}\n{negative_pct:.1f}%", ha="center", va="center", fontsize=17, color="#333333")
    if positive_pct > 0:
        ax.text(0.7, 0, f"{positive_label} {positive_pct:.1f}%", ha="center", va="center", fontsize=17, color="#ffffff")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {os.path.basename(sys.argv[0])} <papers.json>")
        sys.exit(1)

    filename = sys.argv[1]
    papers = load_papers(filename)

    total = 0
    impact_any = 0
    total_explicit = 0

    for paper in papers:
        p = _get_parsed(paper)
        total += 1

        if (p.get("mentions_social_impact") or p.get("mentions_ecological_impact")
                or p.get("mentions_sustainability_impact") or p.get("mentions_un_sdgs")):
            impact_any += 1
        if p.get("motivated_by_sdgs"):
            total_explicit += 1

    if total == 0:
        print("No papers found in input data.")
        sys.exit(1)

    p_impact_any = safe_pct(impact_any, total)
    p_explicit = safe_pct(total_explicit, total)

    print(
        "Mentions: "
        f"{impact_any:,}/{total:,} ({p_impact_any:.1f}%) | "
        "Explicitly motivated: "
        f"{total_explicit:,}/{total:,} ({p_explicit:.1f}%)"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor="#FFFFFF")
    fig.patch.set_facecolor("#FFFFFF")

    draw_binary_donut(
        ax1,
        p_impact_any,
        "Yes",
        "No",
        COLORS["impact_any"],
        f"Mentioning\nBroader\nImpacts?",
        ""#"The Awareness Gap",
    )
    draw_binary_donut(
        ax2,
        p_explicit,
        "Yes",
        "No",
        COLORS["un_sdg"],
        f"Motivated\nby\nSustainability?",
        ""#"The Motivation Gap",
    )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()