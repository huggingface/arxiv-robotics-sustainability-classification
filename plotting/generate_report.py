#!/usr/bin/env python3
"""
Generate a PDF report of all plots for a given papers JSON file.

Outputs:
  <run_dir>/report/
    total_in_time.pdf
    explicit_sustainability_vs_total.pdf
    impacts_mentions_motivation.pdf
    impacts_average_static.pdf
    sdg_absolute.pdf
    sdg_relative.pdf
    README.md

Usage:
  python plotting/generate_report.py <papers.json> [output_run_dir]

If output_run_dir is omitted, the report is saved next to papers.json in a
sibling `report/` directory.  The papers.json may also be a raw results.json
from the inference pipeline; both formats are supported.
"""

import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")  # headless backend for report generation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, Patch
from matplotlib.widgets import Slider

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

COLORS = {
    "no_impact": "#E3E3E3",
    "impact_any": "#4e79a7",
    "social": "#D68B63",
    "ecological": "#B6BB66",
    "sustainability": "#62C1A4",
    "un_sdg": "#59a14f",
    "explicit": "#59a14f",
    "aligned": "#9fa8b3",
    "none_sdg": "#e15759",
    "total_bg": "#E0E0E0",
    "monthly": "#4e79a7",
    "quarterly": "#E0E0E0",
}

SDG_NAMES = [
    "No Poverty", "No Hunger", "Health", "Education", "Gender Eq.",
    "Clean Water", "Clean Energy", "Decent Work", "Industry",
    "Reduce Ineq.", "Sust. Cities", "Consumption", "Climate Action",
    "Life In Water", "Life On Land", "Peace & Justice", "Partnerships",
]


def _get_parsed(paper: dict) -> dict:
    """Pre-parsed classification fields — works for both compiled and raw results.json."""
    if "motivated_by_sdgs" in paper or "mentions_social_impact" in paper:
        return paper
    return (paper.get("deepseek") or {}).get("parsed") or {}


def load_papers(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_dt(published: str) -> Optional[datetime]:
    if not published:
        return None
    try:
        return datetime.fromisoformat(published.replace("Z", "+00:00"))
    except Exception:
        return None


def _quarter(dt: datetime) -> Tuple[int, int]:
    return dt.year, (dt.month - 1) // 3 + 1


def _build_quarter_range(keys) -> List[Tuple[int, int]]:
    if not keys:
        return []
    sy, sq = min(keys)
    ey, eq = max(keys)
    out, y, q = [], sy, sq
    while (y, q) <= (ey, eq):
        out.append((y, q))
        q += 1
        if q > 4:
            q, y = 1, y + 1
    return out


def _sdg_nums(sdg_list) -> Set[int]:
    nums: Set[int] = set()
    for s in (sdg_list or []):
        m = re.search(r"\d+", str(s))
        if m:
            nums.add(int(m.group(0)))
    return nums


def _safe_pct(n: float, d: float) -> float:
    return (n / d * 100.0) if d > 0 else 0.0


def _save(fig: plt.Figure, path: Path, label: str) -> None:
    fig.savefig(path, format="pdf", bbox_inches="tight")
    print(f"  Saved {label}: {path.name}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 1 — total papers in time (monthly + quarterly)
# ---------------------------------------------------------------------------

def _plot_total_in_time(papers: List[dict]) -> plt.Figure:
    monthly: Dict[Tuple[int, int], int] = defaultdict(int)
    quarterly: Dict[Tuple[int, int], int] = defaultdict(int)
    min_dt = datetime.max.replace(tzinfo=timezone.utc)
    max_dt = datetime.min.replace(tzinfo=timezone.utc)

    for paper in papers:
        dt = _parse_dt(paper.get("published"))
        if not dt:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        min_dt = min(min_dt, dt)
        max_dt = max(max_dt, dt)
        monthly[(dt.year, dt.month)] += 1
        quarterly[_quarter(dt)] += 1

    if not monthly:
        return None

    start_abs = min_dt.year * 12 + min_dt.month - 1
    end_abs = max_dt.year * 12 + max_dt.month - 1
    x_months, y_months, jan_idx, jan_lbls = [], [], [], []
    for abs_m in range(start_abs, end_abs + 1):
        yr, mo = abs_m // 12, abs_m % 12 + 1
        idx = abs_m - start_abs
        x_months.append(idx)
        y_months.append(monthly.get((yr, mo), 0))
        if mo == 1:
            jan_idx.append(idx)
            jan_lbls.append(f"Jan {yr}")

    x_q, y_q = [], []
    for (yr, q), cnt in quarterly.items():
        center_mo = (q - 1) * 3 + 2
        x_q.append(yr * 12 + center_mo - 1 - start_abs)
        y_q.append(cnt)

    fig, ax = plt.subplots(figsize=(16, 6), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.bar(x_q, y_q, width=3.0 * 0.82, color=COLORS["quarterly"], label="Quarterly Total", zorder=2)
    ax.bar(x_months, y_months, width=0.75, color=COLORS["monthly"], label="Monthly Total", zorder=3)
    ax.set_xticks(jan_idx)
    ax.set_xticklabels(jan_lbls, rotation=45, ha="right", fontsize=13)
    ax.set_xlim(-1.5, len(x_months) + 0.5)
    ax.tick_params(axis="y", labelsize=13)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", which="both", left=False)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.3, color="#B0B0B0")
    ax.set_axisbelow(True)
    legend_elements = [
        Patch(facecolor=COLORS["quarterly"], label="Quarterly Total"),
        Patch(facecolor=COLORS["monthly"], label="Monthly Total"),
    ]
    ax.legend(handles=legend_elements, ncol=2, frameon=False, fontsize=17, handlelength=1.2)
    ax.set_xlabel("Publication Date", fontsize=17, labelpad=10)
    ax.set_ylabel("Number of Papers Published", fontsize=17, labelpad=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 2 — explicit sustainability vs total (quarterly)
# ---------------------------------------------------------------------------

def _plot_explicit_vs_total(papers: List[dict]) -> plt.Figure:
    totals: Dict[Tuple[int, int], int] = defaultdict(int)
    explicit: Dict[Tuple[int, int], int] = defaultdict(int)

    for paper in papers:
        dt = _parse_dt(paper.get("published"))
        if not dt:
            continue
        qk = _quarter(dt)
        totals[qk] += 1
        if _get_parsed(paper).get("motivated_by_sdgs"):
            explicit[qk] += 1

    period_keys = _build_quarter_range(set(totals.keys()))
    if not period_keys:
        return None

    labels = [f"{y} Q{q}" for y, q in period_keys]
    x = np.arange(len(period_keys))
    total_vals = np.array([totals.get(k, 0) for k in period_keys], dtype=float)
    explicit_vals = np.array([explicit.get(k, 0) for k in period_keys], dtype=float)
    pct = np.divide(explicit_vals, total_vals, out=np.zeros_like(explicit_vals), where=total_vals > 0) * 100

    fig, ax1 = plt.subplots(figsize=(16, 6), facecolor="#FFFFFF")
    ax1.set_facecolor("#FFFFFF")
    ax1.bar(x, total_vals, width=0.82, color=COLORS["total_bg"], label="Total Papers", zorder=1)
    ax1.bar(x, explicit_vals, width=0.82, color=COLORS["explicit"], label="Explicit Sustainability Motivation", zorder=2)
    ax1.set_ylabel("Number of Papers", fontsize=17, labelpad=10)
    ax1.set_xlabel("Year and Quarter", fontsize=17, labelpad=10)
    ax2 = ax1.twinx()
    line, = ax2.plot(x, pct, color="#4e79a7", marker="o", linewidth=1.8, label="Explicit / Total (%)", zorder=3)
    ax2.set_ylabel("Explicitly Motivated Papers (%)", fontsize=17, labelpad=10)
    step = 2 if len(labels) > 14 else 1
    tick_idx = list(range(0, len(labels), step))
    ax1.set_xticks(x[tick_idx])
    ax1.set_xticklabels([labels[i] for i in tick_idx], rotation=45, ha="right", fontsize=13)
    for ax in (ax1, ax2):
        ax.tick_params(axis="y", labelsize=13)
    ax1.legend(
        [ax1.containers[0], ax1.containers[1], line],
        ["Total Papers", "Explicit Sustainability Motivation", "Explicit / Total (%)"],
        loc="upper center", ncol=3, frameon=False, fontsize=15,
    )
    ax1.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.3, color="#B0B0B0")
    ax1.set_axisbelow(True)
    ax2.grid(False)
    ax1.set_xlim(-0.7, len(x) - 0.3)
    for spine in ("top", "right", "left"):
        ax1.spines[spine].set_visible(False)
    ax1.tick_params(axis="y", which="both", left=False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 3 — impact mentions + motivation donut pair
# ---------------------------------------------------------------------------

def _plot_impact_donuts(papers: List[dict]) -> plt.Figure:
    total = impact_any = explicit = 0
    for paper in papers:
        p = _get_parsed(paper)
        total += 1
        if any(p.get(k) for k in ("mentions_social_impact", "mentions_ecological_impact",
                                   "mentions_sustainability_impact", "mentions_un_sdgs")):
            impact_any += 1
        if p.get("motivated_by_sdgs"):
            explicit += 1

    if total == 0:
        return None

    p_any = _safe_pct(impact_any, total)
    p_exp = _safe_pct(explicit, total)

    def _donut(ax, pct, center_txt, pos_color):
        angle = pct / 100 * 360
        ax.pie(
            [pct, 100 - pct],
            colors=[pos_color, COLORS["no_impact"]],
            startangle=angle / 2,
            counterclock=False,
            wedgeprops=dict(width=0.6, edgecolor="white", linewidth=1.5),
        )
        ax.text(0, 0, center_txt, ha="center", va="center", fontsize=17, color="#333333")
        if 100 - pct > 0:
            ax.text(-0.7, 0, f"No\n{100 - pct:.1f}%", ha="center", va="center", fontsize=15, color="#666666")
        if pct > 0:
            ax.text(0.7, 0, f"Yes\n{pct:.1f}%", ha="center", va="center", fontsize=15, color="white")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor="#FFFFFF")
    fig.patch.set_facecolor("#FFFFFF")
    _donut(ax1, p_any, "Mentioning\nBroader\nImpacts?", COLORS["impact_any"])
    _donut(ax2, p_exp, "Motivated\nby\nSustainability?", COLORS["un_sdg"])
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 4 — impact breakdown donut + horizontal bars
# ---------------------------------------------------------------------------

def _plot_impact_breakdown(papers: List[dict]) -> plt.Figure:
    total = impact_any = social = ecological = sustainability = un_sdgs = 0
    for paper in papers:
        p = _get_parsed(paper)
        total += 1
        s = bool(p.get("mentions_social_impact"))
        e = bool(p.get("mentions_ecological_impact"))
        su = bool(p.get("mentions_sustainability_impact"))
        u = bool(p.get("mentions_un_sdgs"))
        if s: social += 1
        if e: ecological += 1
        if su: sustainability += 1
        if u: un_sdgs += 1
        if s or e or su or u: impact_any += 1

    if total == 0:
        return None

    p_any = _safe_pct(impact_any, total)
    p_no = 100 - p_any

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 5.5), facecolor="#FFFFFF",
                                    gridspec_kw={"width_ratios": [1, 1.3]})
    fig.patch.set_facecolor("#FFFFFF")

    angle = p_any / 100 * 360
    ax1.pie([p_any, p_no], colors=[COLORS["impact_any"], COLORS["no_impact"]],
            startangle=angle / 2, counterclock=False,
            wedgeprops=dict(width=0.4, edgecolor="white", linewidth=1.5))
    ax1.text(0, 0, f"Total Papers\n{total:,}", ha="center", va="center", fontsize=15, color="#333333")
    ax1.text(-0.8, 0, f"No\nImpact\n{p_no:.1f}%", ha="center", va="center", fontsize=14, color="#666666")
    ax1.text(np.cos(0) * 0.8, np.sin(0) * 0.8, f"Any\nImpact\n{p_any:.1f}%",
             ha="center", va="center", fontsize=12, color="white")

    bar_cats = ["UN SDGs", "Sustainability", "Ecological", "Social"]
    bar_pcts = [_safe_pct(un_sdgs, total), _safe_pct(sustainability, total),
                _safe_pct(ecological, total), _safe_pct(social, total)]
    bar_counts = [un_sdgs, sustainability, ecological, social]
    bar_colors = [COLORS["un_sdg"], COLORS["sustainability"], COLORS["ecological"], COLORS["social"]]
    y_pos = np.arange(len(bar_cats))
    bars = ax2.barh(y_pos, bar_pcts, color=bar_colors, height=0.75, edgecolor="white", linewidth=1.2)
    for i, bar in enumerate(bars):
        w = bar.get_width()
        fmt = f"{w:.2f}%" if w < 1 else f"{w:.1f}%"
        ax2.text(w, bar.get_y() + bar.get_height() / 2,
                 f"  {bar_counts[i]:,} papers\n  ({fmt})", ha="left", va="center", fontsize=14)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(bar_cats, fontsize=15)
    ax2.set_xlabel("Share of Total Papers (%)", fontsize=15, labelpad=10)
    ax2.set_xlim(0, max(bar_pcts) * 1.35 if bar_pcts else 10)
    ax2.set_ylim(-0.6, len(bar_cats) - 0.4)
    for spine in ("top", "right", "bottom"):
        ax2.spines[spine].set_visible(False)
    ax2.xaxis.grid(True, linestyle="--", alpha=0.3, color="#B0B0B0")
    ax2.set_axisbelow(True)

    con1 = ConnectionPatch(
        xyA=(np.cos(np.deg2rad(angle / 2)), np.sin(np.deg2rad(angle / 2))), coordsA=ax1.transData,
        xyB=(0, 3.4), coordsB=ax2.transData, color=COLORS["impact_any"],
        linestyle="--", linewidth=1.5, alpha=0.6, connectionstyle="arc3,rad=-0.2")
    con2 = ConnectionPatch(
        xyA=(np.cos(np.deg2rad(-angle / 2)), np.sin(np.deg2rad(-angle / 2))), coordsA=ax1.transData,
        xyB=(0, -0.4), coordsB=ax2.transData, color=COLORS["impact_any"],
        linestyle="--", linewidth=1.5, alpha=0.6, connectionstyle="arc3,rad=0.2")
    fig.add_artist(con1)
    fig.add_artist(con2)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 6 & 7 — SDG absolute + relative bar charts
# ---------------------------------------------------------------------------

def _build_sdg_counts(papers: List[dict]):
    explicit_counts = {i: 0 for i in range(1, 18)}
    indirect_counts = {i: 0 for i in range(1, 18)}
    no_sdgs = 0

    for paper in papers:
        p = _get_parsed(paper)
        exp = _sdg_nums(p.get("motivated_by_sdgs"))
        ind = _sdg_nums(p.get("aligned_with_sdgs"))
        if not exp and not ind:
            no_sdgs += 1
            continue
        for sdg_id in exp:
            if 1 <= sdg_id <= 17:
                explicit_counts[sdg_id] += 1
                indirect_counts[sdg_id] += 1
        for sdg_id in ind:
            if 1 <= sdg_id <= 17 and sdg_id not in exp:
                indirect_counts[sdg_id] += 1

    sorted_ids = sorted(range(1, 18), key=lambda i: indirect_counts[i])
    labels = ["No SDGs"] + [f"SDG {i}: {SDG_NAMES[i - 1]}" for i in sorted_ids]
    explicit = np.array([0] + [explicit_counts[i] for i in sorted_ids], dtype=float)
    aligned = np.array([0] + [indirect_counts[i] for i in sorted_ids], dtype=float)
    none = np.array([no_sdgs] + [0] * 17, dtype=float)
    return labels, explicit, aligned, none, len(papers)


def _plot_sdg_absolute(labels, explicit, aligned, none, n_papers) -> plt.Figure:
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 9), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.barh(y, aligned, height=0.72, color=COLORS["aligned"], alpha=0.45, label="SDG Aligned", zorder=1)
    ax.barh(y, explicit, height=0.72, color=COLORS["explicit"], label="SDG Motivated", zorder=2)
    ax.barh(y, none, height=0.72, color=COLORS["none_sdg"], label="No SDG Relevance", zorder=3)
    for i, (lbl, a_val) in enumerate(zip(labels, aligned)):
        if lbl == "No SDGs" or n_papers <= 0 or a_val <= 0:
            continue
        ax.text(a_val + max(aligned) * 0.01, i, f"{a_val / n_papers * 100:.1f}%",
                va="center", fontsize=11, color="#333333")
    for i, n_val in enumerate(none):
        if n_papers <= 0 or n_val <= 0:
            continue
        ax.text(n_val + max(aligned) * 0.01, i, f"{n_val / n_papers * 100:.1f}%",
                va="center", fontsize=11, color=COLORS["none_sdg"])
    ax.set_xlabel("Number of Papers", fontsize=13, labelpad=10)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlim(0, float(np.max(np.vstack([explicit, aligned, none]))) * 1.3 + 1)
    ax.grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.3, color="#B0B0B0")
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", which="both", left=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


def _plot_sdg_relative(labels, explicit, aligned) -> plt.Figure:
    labels_sdg = labels[1:]
    exp_sdg = explicit[1:]
    aln_sdg = aligned[1:]
    aln_only = np.maximum(aln_sdg - exp_sdg, 0)
    exp_rel = np.divide(exp_sdg, aln_sdg, out=np.zeros_like(exp_sdg), where=aln_sdg > 0) * 100
    aln_rel = np.divide(aln_only, aln_sdg, out=np.zeros_like(aln_only), where=aln_sdg > 0) * 100
    idx = np.argsort(exp_rel)
    labels_s = [labels_sdg[i] for i in idx]
    exp_s = exp_rel[idx]
    aln_s = aln_rel[idx]
    y = np.arange(len(labels_s))
    fig, ax = plt.subplots(figsize=(12, 9), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.barh(y, exp_s, height=0.72, color=COLORS["explicit"], label="SDG Motivated", zorder=2)
    for i, val in enumerate(exp_s):
        if val > 50:
            ax.text(val / 2, i, f"{val:.1f}%", va="center", ha="center", fontsize=10, color="white")
        else:
            ax.text(val + 1, i, f"{val:.1f}%", va="center", ha="left", fontsize=10, color="#333333")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share Within SDG (%)", fontsize=13, labelpad=10)
    ax.set_yticks(y)
    ax.set_yticklabels(labels_s, fontsize=12)
    ax.grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.3, color="#B0B0B0")
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", which="both", left=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------

def _write_readme(report_dir: Path, papers_path: str, n_papers: int,
                  plot_files: List[Tuple[str, str, str]]) -> None:
    lines = [
        "# SDG Classification Report",
        "",
        f"**Source:** `{papers_path}`  ",
        f"**Papers analysed:** {n_papers:,}  ",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "---",
        "",
        "## Plots",
        "",
    ]
    for fname, title, desc in plot_files:
        lines += [
            f"### {title}",
            "",
            desc,
            "",
            f"![{title}]({fname})",
            "",
        ]
    (report_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved README: README.md")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PLOT_META = [
    ("total_in_time.pdf", "Total Papers Over Time",
     "Monthly (blue) and quarterly (grey) paper counts across the full date range."),
    ("explicit_sustainability_vs_total.pdf", "Explicit Sustainability Motivation Over Time",
     "Quarterly total papers (grey) overlaid with papers explicitly motivated by sustainability (green), "
     "and the percentage trend line (blue)."),
    ("impacts_mentions_motivation.pdf", "Impact Mentions vs Sustainability Motivation",
     "Two donuts: left shows share of papers mentioning any broader impact; "
     "right shows share explicitly motivated by sustainability goals."),
    ("impacts_breakdown.pdf", "Impact Mention Breakdown",
     "Donut (any impact vs none) linked to a horizontal bar chart showing the individual overlap counts "
     "for social, ecological, sustainability, and UN SDG mentions."),
    ("sdg_absolute.pdf", "SDG Coverage — Absolute Counts",
     "Horizontal bars per SDG showing papers explicitly motivated by that SDG (dark green) and "
     "papers aligned with it regardless of motivation (grey, overlaid)."),
    ("sdg_relative.pdf", "SDG Coverage — Relative Share",
     "For each SDG, the percentage of SDG-mentioning papers that are explicitly motivated by it."),
]


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {os.path.basename(sys.argv[0])} <papers.json> [run_dir]")
        sys.exit(1)

    papers_path = sys.argv[1]
    if not os.path.exists(papers_path):
        print(f"File not found: {papers_path}")
        sys.exit(1)

    papers = load_papers(papers_path)
    print(f"Loaded {len(papers)} papers from {papers_path}")

    # Resolve report directory
    if len(sys.argv) >= 3:
        report_dir = Path(sys.argv[2]) / "report"
    else:
        report_dir = Path(papers_path).parent / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    print(f"Report directory: {report_dir}")

    generators = [
        _plot_total_in_time,
        _plot_explicit_vs_total,
        _plot_impact_donuts,
        _plot_impact_breakdown
    ]
    sdg_labels, sdg_explicit, sdg_aligned, sdg_none, n = _build_sdg_counts(papers)

    figs = [g(papers) for g in generators]
    figs.append(_plot_sdg_absolute(sdg_labels, sdg_explicit, sdg_aligned, sdg_none, n))
    figs.append(_plot_sdg_relative(sdg_labels, sdg_explicit, sdg_aligned))

    saved = []
    for fig, (fname, title, desc) in zip(figs, PLOT_META):
        if fig is None:
            print(f"  Skipped {fname} (no data)")
            continue
        _save(fig, report_dir / fname, title)
        saved.append((fname, title, desc))

    _write_readme(report_dir, papers_path, len(papers), saved)
    print(f"\nReport complete: {report_dir}")


if __name__ == "__main__":
    main()
