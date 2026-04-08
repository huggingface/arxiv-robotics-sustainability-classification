#!/usr/bin/env python3
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from matplotlib.patches import Patch

import matplotlib.pyplot as plt
import numpy as np

# Define colors matching the clean aesthetic of the image
QUARTERLY_COLOR = '#E0E0E0'  # Light gray for the background
MONTHLY_COLOR = '#4e79a7'    # Solid blue for the foreground


def load_papers(filename: str):
    """Loads the raw papers JSON file."""
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


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {os.path.basename(sys.argv[0])} <papers.json>")
        sys.exit(1)

    filename = sys.argv[1]
    papers = load_papers(filename)

    # 1. Aggregate the data
    monthly_counts = defaultdict(int)
    quarterly_counts = defaultdict(int)

    from datetime import timezone

    min_date = datetime.max.replace(tzinfo=timezone.utc)
    max_date = datetime.min.replace(tzinfo=timezone.utc)

    for paper in papers:
        pub_date_str = paper.get("published")
        if not pub_date_str:
            continue
        try:
            # Handle ISO formats
            dt = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        
        if dt < min_date: min_date = dt
        if dt > max_date: max_date = dt

        month_key = (dt.year, dt.month)
        quarter = (dt.month - 1) // 3 + 1
        quarter_key = (dt.year, quarter)

        monthly_counts[month_key] += 1
        quarterly_counts[quarter_key] += 1

    if not monthly_counts:
        print("No valid date data found.")
        sys.exit(1)

    # 2. Build continuous timeline
    start_abs_month = min_date.year * 12 + min_date.month - 1
    end_abs_month = max_date.year * 12 + max_date.month - 1

    x_months = []
    y_months = []
    month_labels = []
    
    # Store indices where January occurs for x-axis ticks
    jan_indices = []
    jan_labels = []

    for abs_m in range(start_abs_month, end_abs_month + 1):
        yr = abs_m // 12
        mo = abs_m % 12 + 1
        
        plot_idx = abs_m - start_abs_month
        x_months.append(plot_idx)
        y_months.append(monthly_counts.get((yr, mo), 0))
        month_labels.append((yr, mo))
        
        if mo == 1:
            jan_indices.append(plot_idx)
            jan_labels.append(f"Jan {yr}")

    # 3. Calculate exact center for Quarterly Bars
    # A quarter spans 3 months. By finding the "center month" of a quarter,
    # we can map the quarterly bar directly over its 3 monthly bars.
    x_quarters = []
    y_quarters = []

    for (yr, q), count in quarterly_counts.items():
        center_mo = (q - 1) * 3 + 2  # Q1 center is Feb (2), Q2 is May (5), etc.
        center_abs_m = yr * 12 + center_mo - 1
        plot_idx = center_abs_m - start_abs_month
        
        x_quarters.append(plot_idx)
        y_quarters.append(count)

    # ==========================================
    # Plotting
    # ==========================================
    fig, ax = plt.subplots(figsize=(16, 6), facecolor='#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    # Plot Background (Quarterly Data)
    # width=3.0 perfectly covers the 3 months making up the quarter
    ax.bar(x_quarters, y_quarters, width=3.0*0.82, color=QUARTERLY_COLOR, label="Quarterly Total", zorder=2)
    
    # Plot Foreground (Monthly Data)
    ax.bar(x_months, y_months, width=0.75, color=MONTHLY_COLOR, label="Monthly Total", zorder=3)

    # Styling Axis
    ax.set_xticks(jan_indices)
    ax.set_xticklabels(jan_labels, rotation=45, ha='right', fontsize=13, color='#111111')
    ax.tick_params(axis='y', labelsize=13, color='#111111')
    
    # Expand x-axis slightly so edge bars aren't cut off
    ax.set_xlim(-1.5, len(x_months) + 0.5)

    # Clean borders (Spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False) # Hide left line to match the floating look
    ax.spines['bottom'].set_color('#333333')
    
    # Remove y-axis tick marks (the little lines) but keep the labels
    ax.tick_params(axis='y', which='both', left=False)

    # Add dark, solid horizontal grid lines beneath the bars
    ax.grid(axis='y', linestyle="--", linewidth=0.8, alpha=0.3, color="#B0B0B0")
    
    ax.set_axisbelow(True) # Ensure grid is behind the bars

    # Custom Centered Top Legend
    legend_elements = [
        Patch(facecolor=QUARTERLY_COLOR, label='Quarterly Total'),
        Patch(facecolor=MONTHLY_COLOR, label='Monthly Total')
    ]
    ax.legend(handles=legend_elements, ncol=2, frameon=False, fontsize=17, handlelength=1.2, handleheight=1.2)

    
    plt.xlabel("Publication Date", fontsize=17, color='#333333', labelpad=10)
    plt.ylabel("Number of Papers Published", fontsize=17, color='#333333', labelpad=10)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()