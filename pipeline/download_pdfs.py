#!/usr/bin/env python3
"""
Download arXiv PDFs for papers in a metadata JSON file.

Papers can optionally be filtered by date range. Files that already exist
on disk are skipped automatically.

Adapted from download_pdfs_multithread.py.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests


DEFAULT_WORKERS = 8
DEFAULT_TIMEOUT = 30


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download PDFs from arXiv metadata JSON.")
    p.add_argument("metadata_json", help="Path to arXiv metadata JSON file.")
    p.add_argument("--output-folder", required=True, help="Folder to save PDFs into.")
    p.add_argument("--start-date", default="", help="Only download papers on or after this date (YYYY-MM-DD).")
    p.add_argument("--end-date", default="", help="Only download papers on or before this date (YYYY-MM-DD).")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"Parallel download threads (default: {DEFAULT_WORKERS}).")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help=f"HTTP timeout per request in seconds (default: {DEFAULT_TIMEOUT}).")
    return p.parse_args()


def _arxiv_id_to_filename(arxiv_id: str) -> str:
    return arxiv_id.split("/")[-1].replace(":", "_") + ".pdf"


def _filter_by_date(entries: list, start_date: str, end_date: str) -> list:
    if not start_date and not end_date:
        return entries
    start_dt = datetime.fromisoformat(start_date) if start_date else datetime.min
    end_dt = datetime.fromisoformat(end_date) if end_date else datetime.max
    filtered = [
        e for e in entries
        if start_dt <= datetime.fromisoformat(e["published"].split("T")[0]) <= end_dt
    ]
    print(f"Filtered {len(filtered)} / {len(entries)} entries by date range.")
    return filtered


def _download_one(entry: dict, output_folder: str, timeout: int):
    pdf_url = entry.get("pdf_url")
    if not pdf_url:
        return entry["id"], False, "no pdf_url"

    filename = _arxiv_id_to_filename(entry["id"])
    save_path = os.path.join(output_folder, filename)

    if os.path.exists(save_path):
        return entry["id"], True, "already exists"

    try:
        r = requests.get(pdf_url, timeout=timeout)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(r.content)
        return entry["id"], True, "downloaded"
    except Exception as e:
        return entry["id"], False, str(e)


def download_pdfs(
    metadata_json: str,
    output_folder: str,
    start_date: str = "",
    end_date: str = "",
    workers: int = DEFAULT_WORKERS,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict:
    """
    Download PDFs for papers in *metadata_json* to *output_folder*.
    Returns a dict with keys 'downloaded', 'skipped', 'failed'.
    """
    os.makedirs(output_folder, exist_ok=True)

    with open(metadata_json, "r", encoding="utf-8") as f:
        entries = json.load(f)
    print(f"Loaded {len(entries)} entries from '{metadata_json}'.")

    entries = _filter_by_date(entries, start_date, end_date)
    print(f"Starting downloads for {len(entries)} papers with {workers} threads ...")

    stats = {"downloaded": 0, "skipped": 0, "failed": 0}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_download_one, e, output_folder, timeout): e
            for e in entries
        }
        for future in as_completed(futures):
            arxiv_id, success, msg = future.result()
            if success:
                if msg == "downloaded":
                    stats["downloaded"] += 1
                    print(f"  [OK]   {arxiv_id}")
                else:
                    stats["skipped"] += 1
            else:
                stats["failed"] += 1
                print(f"  [FAIL] {arxiv_id}: {msg}")

    print(
        f"\nSummary — downloaded: {stats['downloaded']}  "
        f"skipped: {stats['skipped']}  failed: {stats['failed']}"
    )
    return stats


def main() -> None:
    args = parse_args()
    download_pdfs(
        metadata_json=args.metadata_json,
        output_folder=args.output_folder,
        start_date=args.start_date,
        end_date=args.end_date,
        workers=args.workers,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
