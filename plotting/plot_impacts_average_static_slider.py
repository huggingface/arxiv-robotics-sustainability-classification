#!/usr/bin/env python3
import json
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch

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


def parse_year(published: str):
    if not published:
        return None
    try:
        dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
        return dt.year
    except Exception:
        return None


def safe_pct(value: int, total: int) -> float:
    return (100.0 * value / total) if total > 0 else 0.0


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {os.path.basename(sys.argv[0])} <papers.json>")
        sys.exit(1)

    filename = sys.argv[1]
    papers = load_papers(filename)

    total = 0
    impact_any = 0
    total_social = 0
    total_ecological = 0
    total_sustainability = 0
    total_un_sdgs = 0

    years = []

    for paper in papers:
        p = _get_parsed(paper)
        total += 1

        y = parse_year(paper.get("published"))
        if y is not None:
            years.append(y)

        # Overlapping counts
        if p.get("mentions_social_impact"): total_social += 1
        if p.get("mentions_ecological_impact"): total_ecological += 1
        if p.get("mentions_sustainability_impact"): total_sustainability += 1
        if p.get("mentions_un_sdgs"): total_un_sdgs += 1
        if (p.get("mentions_social_impact") or p.get("mentions_ecological_impact")
                or p.get("mentions_sustainability_impact") or p.get("mentions_un_sdgs")):
            impact_any += 1

    if total == 0:
        print("No papers found in input data.")
        sys.exit(1)

    no_impact = total - impact_any

    # Percentages
    p_no_impact = safe_pct(no_impact, total)
    p_impact_any = safe_pct(impact_any, total)
    p_social = safe_pct(total_social, total)
    p_ecological = safe_pct(total_ecological, total)
    p_sustainability = safe_pct(total_sustainability, total)
    p_un_sdgs = safe_pct(total_un_sdgs, total)

    # ==========================================
    # Zoom-In Plotting Logic (Donut -> Bar Chart)
    # ==========================================
    # Changed facecolor to pure white '#FFFFFF'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 5.5), facecolor='#FFFFFF', gridspec_kw={'width_ratios': [1, 1.3]})
    fig.patch.set_facecolor('#FFFFFF')

    # --- LEFT SIDE: THE MACRO DONUT (Mutually Exclusive) ---
    macro_labels = ['Any Impact Mention', 'No Impact Mention']
    macro_sizes = [p_impact_any, p_no_impact]
    macro_colors = [COLORS["impact_any"], COLORS["no_impact"]]

    # Center the tiny "Impact Any" slice exactly on the right side (0 degrees)
    angle = (p_impact_any / 100) * 360
    start_angle = (angle / 2)

    wedges, texts = ax1.pie(
        macro_sizes, 
        colors=macro_colors, 
        startangle=start_angle, 
        counterclock=False,
        wedgeprops=dict(width=0.4, edgecolor='white', linewidth=1.5)
    )

    # Center Text for Donut
    ax1.text(0, 0, f"Total Papers\n{total:,}", ha='center', va='center', fontsize=15, color='#333333')
    #ax1.set_title("The Awareness Gap", fontsize=14, color='#333333', pad=15)

    # Add text to Donut slices
    ax1.text(-0.8, 0, f"No\nImpact\n{p_no_impact:.1f}%", ha='center', va='center', fontsize=14, color='#666666')
    
    rad = np.deg2rad(0) 
    ax1.text(np.cos(rad)*0.8, np.sin(rad)*0.8, f"Any\nImpact\n{p_impact_any:.1f}%", 
             ha='center', va='center', fontsize=12, color='white')


    # --- RIGHT SIDE: THE MICRO BAR CHART (Overlapping) ---
    bar_categories = ['UN SDGs', 'Sustainability', 'Ecological', 'Social']
    bar_percentages = [p_un_sdgs, p_sustainability, p_ecological, p_social]
    bar_counts = [total_un_sdgs, total_sustainability, total_ecological, total_social]
    bar_colors = [COLORS["un_sdg"], COLORS["sustainability"], COLORS["ecological"], COLORS["social"]]

    y_pos = np.arange(len(bar_categories))
    
    # Draw bars
    bars = ax2.barh(y_pos, bar_percentages, color=bar_colors, height=0.75, edgecolor='white', linewidth=1.2)

    # Add data labels outside the bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        format_pct = f"{width:.2f}%" if width < 1 else f"{width:.1f}%"
        label_text = f"  {bar_counts[i]:,} papers\n  ({format_pct})"
        ax2.text(width, bar.get_y() + bar.get_height()/2, label_text, 
                 ha='left', va='center', fontsize=15, color='#333333')

    # Formatting Bar Chart
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(bar_categories, fontsize=15, color='#333333')
    ax2.set_xlabel('Share of Total Papers (%)', fontsize=15, color='#333333', labelpad=10)
    ax2.set_xlim(0, max(bar_percentages) * 1.3)
    
    # Set explicitly expanded Y-limits so the dashed lines route well above/below the text!
    ax2.set_ylim(-0.6, len(bar_categories) - 0.4)
    
    #ax2.set_title("Breakdown of Specific Impact Mentions (Note: Categories can overlap)", 
    #              fontsize=12, color='#4A4A4A', pad=15, loc='left')

    # Clean up spines and add subtle grid
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#B0B0B0')
    ax2.spines['bottom'].set_visible(False)
    ax2.xaxis.grid(True, linestyle='--', alpha=0.3, color='#B0B0B0')
    ax2.set_axisbelow(True)

    # --- DRAW SPLINE CONNECTION LINES (The "Zoom" Effect) ---
    # Using arc3 with a positive radius makes the top line bow upwards, avoiding the "Social" text
    con1 = ConnectionPatch(xyA=(np.cos(np.deg2rad(angle/2)), np.sin(np.deg2rad(angle/2))), coordsA=ax1.transData, 
                           xyB=(0, 3.4), coordsB=ax2.transData, color=COLORS["impact_any"], 
                           linestyle="--", linewidth=1.5, alpha=0.6,
                           connectionstyle="arc3,rad=-0.2")
    
    # Using arc3 with a negative radius makes the bottom line bow downwards, avoiding the "UN SDGs" text
    con2 = ConnectionPatch(xyA=(np.cos(np.deg2rad(-angle/2)), np.sin(np.deg2rad(-angle/2))), coordsA=ax1.transData, 
                           xyB=(0, -0.4), coordsB=ax2.transData, color=COLORS["impact_any"], 
                           linestyle="--", linewidth=1.5, alpha=0.6,
                           connectionstyle="arc3,rad=0.2")
    
    fig.add_artist(con1)
    fig.add_artist(con2)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()