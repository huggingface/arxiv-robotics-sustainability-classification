#!/usr/bin/env python3
"""
Prepare metadata: filter a full arXiv metadata JSON to only the papers
that have a local PDF, and enrich each record with a `pdf_path` field.

Adapted from remap_pdfs_metadata.py.
"""

import argparse
import json
import os


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Filter metadata to locally available PDFs and add pdf_path field."
    )
    p.add_argument("metadata_json", help="Full arXiv metadata JSON file.")
    p.add_argument("pdf_folder", help="Folder containing downloaded PDF files.")
    p.add_argument(
        "--output",
        default="",
        help="Output path (default: <pdf_folder>/metadata_for_pdfs.json).",
    )
    return p.parse_args()


def _arxiv_id_to_stem(arxiv_id: str) -> str:
    return arxiv_id.split("/")[-1].replace(":", "_")


def prepare_metadata(
    metadata_json: str,
    pdf_folder: str,
    output: str = "",
) -> list:
    """
    Filter *metadata_json* to entries with a matching PDF in *pdf_folder*,
    add an absolute `pdf_path` field, and write the result to *output*.
    Returns the list of matched entries.
    """
    if not output:
        output = os.path.join(pdf_folder, "metadata_for_pdfs.json")

    with open(metadata_json, "r", encoding="utf-8") as f:
        entries = json.load(f)
    print(f"Loaded {len(entries)} entries from '{metadata_json}'.")

    # Build a stem → filename index of local PDFs.
    local_pdfs = {
        f.rsplit(".", 1)[0]: f
        for f in os.listdir(pdf_folder)
        if f.lower().endswith(".pdf")
    }
    print(f"Found {len(local_pdfs)} PDFs in '{pdf_folder}'.")

    matched = []
    for entry in entries:
        stem = _arxiv_id_to_stem(entry["id"])
        if stem in local_pdfs:
            entry = dict(entry)
            entry["pdf_path"] = os.path.abspath(
                os.path.join(pdf_folder, local_pdfs[stem])
            )
            matched.append(entry)

    unmatched_count = len(local_pdfs) - len(matched)
    print(f"Matched {len(matched)} entries. Unmatched PDFs: {unmatched_count}.")

    tmp = output + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(matched, f, indent=2, ensure_ascii=False)
    os.replace(tmp, output)
    print(f"Saved to '{output}'.")
    return matched


def main() -> None:
    args = parse_args()
    prepare_metadata(args.metadata_json, args.pdf_folder, args.output)


if __name__ == "__main__":
    main()
