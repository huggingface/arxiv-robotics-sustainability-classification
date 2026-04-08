#!/usr/bin/env python3
"""
Fetch arXiv metadata (cs.RO by default) for a date range via the arXiv API.

Output is a single JSON array. The file is written atomically after every
batch so interrupted runs simply resume from the existing file on next run.

Adapted from database_download.py.
"""

import argparse
import json
import os
import time
from datetime import datetime

import feedparser


DEFAULT_CATEGORY = "cs.RO"
DEFAULT_OUTPUT = "metadata.json"
BASE_URL = "https://export.arxiv.org/api/query?"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch arXiv metadata for a date range."
    )
    p.add_argument("--start-date", required=True, help="Oldest date to include (YYYY-MM-DD, inclusive).")
    p.add_argument("--end-date", required=True, help="Newest date to include (YYYY-MM-DD, inclusive).")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Output JSON file (default: {DEFAULT_OUTPUT}).")
    p.add_argument("--category", default=DEFAULT_CATEGORY, help=f"arXiv category (default: {DEFAULT_CATEGORY}).")
    p.add_argument("--max-results", type=int, default=50000, help="Maximum papers to collect (default: 50000).")
    p.add_argument("--batch-size", type=int, default=500, help="API page size (default: 500).")
    return p.parse_args()


def _parse_date(s: str) -> datetime:
    return datetime.fromisoformat(s.split("T")[0])


def _entry_published_raw(entry) -> str:
    # Some arXiv feed items can miss `published`; fall back to `updated`.
    return (entry.get("published") or entry.get("updated") or "").strip()


def _entry_id(entry) -> str:
    return (entry.get("id") or "").strip()


def _extract_pdf_url(entry) -> str:
    for link in entry.get("links", []):
        if link.rel == "related" and link.type == "application/pdf":
            return link.href
    return ""


def _load_existing(output_file: str) -> list:
    if not os.path.exists(output_file):
        return []
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"Warning: could not load existing file ({e}). Starting fresh.")
        return []


def _save(output_file: str, entries: list) -> None:
    """Atomic write: write to .tmp then rename."""
    tmp = output_file + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    os.replace(tmp, output_file)


def fetch_metadata(
    start_date: str,
    end_date: str,
    output: str = DEFAULT_OUTPUT,
    category: str = DEFAULT_CATEGORY,
    max_results: int = 50000,
    batch_size: int = 500,
) -> list:
    """
    Download arXiv metadata for papers published between start_date and end_date
    (both inclusive). Saves incrementally to *output* after each page.
    Returns the full list of collected entries.
    """
    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)
    if start_dt > end_dt:
        raise ValueError("start_date must be <= end_date")

    # Heuristic requested by user:
    # if requested window is older than the midpoint between 2015 and today,
    # crawl from oldest papers first (ascending), otherwise newest first.
    anchor_dt = datetime(2015, 1, 1)
    now_dt = datetime.today()
    midpoint_dt = anchor_dt + (now_dt - anchor_dt) / 2
    window_mid_dt = start_dt + (end_dt - start_dt) / 2
    sort_order = "ascending" if window_mid_dt <= midpoint_dt else "descending"

    all_entries = _load_existing(output)
    existing_ids = {e.get("id") for e in all_entries if e.get("id")}
    print(f"Loaded {len(all_entries)} existing entries from '{output}'.")
    print(
        f"Fetch mode: {sort_order} "
        f"(window_mid={window_mid_dt.date()}, pivot={midpoint_dt.date()})"
    )

    api_offset = 0          # position in arXiv's sorted feed
    current_batch = batch_size
    retries_left = 10

    while len(all_entries) < max_results:
        remaining = max_results - len(all_entries)
        this_batch = min(current_batch, remaining)

        query = (
            f"search_query=cat:{category}"
            f"&start={api_offset}"
            f"&max_results={this_batch}"
            f"&sortBy=submittedDate"
            f"&sortOrder={sort_order}"
        )
        print(f"[fetch] offset={api_offset}  batch={this_batch}  collected={len(all_entries)} ...")
        feed = feedparser.parse(BASE_URL + query)

        if not feed.entries:
            retries_left -= 1
            current_batch = current_batch // 2

            if current_batch <= 0 or retries_left <= 0:
                # Do not stop on empty pages; skip forward and keep crawling.
                api_offset += 2
                retries_left = 10
                current_batch = batch_size
                print(
                    "  Empty page after retries; skipping 2 entries "
                    f"(new offset={api_offset}) and continuing ..."
                )
            else:
                print(f"  Empty page — retrying (retries_left={retries_left}, batch={current_batch}) ...")

            time.sleep(0.2)
            continue

        retries_left = 10
        current_batch = batch_size
        added = 0
        stop_early = False
        skipped_malformed = 0

        for entry in feed.entries:
            entry_id = _entry_id(entry)
            published_raw = _entry_published_raw(entry)

            if not entry_id or not published_raw:
                skipped_malformed += 1
                continue

            try:
                pub_dt = _parse_date(published_raw)
            except ValueError:
                skipped_malformed += 1
                continue

            if sort_order == "descending":
                if pub_dt > end_dt:
                    continue                      # paper is too new — keep paging older
                if pub_dt < start_dt:
                    stop_early = True             # reached older-than-window papers
                    break
            else:
                if pub_dt < start_dt:
                    continue                      # paper is too old — keep paging newer
                if pub_dt > end_dt:
                    stop_early = True             # reached newer-than-window papers
                    break
            if entry_id in existing_ids:
                continue

            metadata = {
                "id": entry_id,
                "title": (entry.get("title") or "").strip(),
                "authors": [a.name for a in entry.get("authors", [])],
                "published": published_raw,
                "summary": (entry.get("summary") or "").strip(),
                "pdf_url": _extract_pdf_url(entry),
            }
            all_entries.append(metadata)
            existing_ids.add(entry_id)
            added += 1
            if len(all_entries) >= max_results:
                break

        print(f"  Added {added} entries this page.")
        if skipped_malformed:
            print(f"  Skipped {skipped_malformed} malformed entries (missing id/published).")
        _save(output, all_entries)

        if stop_early or len(all_entries) >= max_results:
            break

        api_offset += this_batch

    print(f"Total collected: {len(all_entries)}. Saved to '{output}'.")
    return all_entries


def main() -> None:
    args = parse_args()
    fetch_metadata(
        start_date=args.start_date,
        end_date=args.end_date,
        output=args.output,
        category=args.category,
        max_results=args.max_results,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
