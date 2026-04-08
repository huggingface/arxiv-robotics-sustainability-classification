#!/usr/bin/env python3
"""
Full pipeline entrypoint.

Runs all three stages in sequence:
  1. fetch_metadata   — download arXiv metadata for a date range
  2. download_pdfs    — download PDFs for the fetched papers
  3. infer_deepseek   — run DeepSeek-V3.2-Exp and save structured results

Each stage writes its output to a predictable location inside --work-dir so
interrupted runs can be safely resumed by re-running this script.

Usage example:
    python run_pipeline.py \\
        --start-date 2025-01-01 --end-date 2025-06-30 \\
        --work-dir ./run/2025-h1 \\
        --hf-token $HF_TOKEN
"""

import argparse
import os
import sys

from pipeline.fetch_metadata import fetch_metadata
from pipeline.download_pdfs import download_pdfs
from pipeline.prepare_metadata import prepare_metadata
from pipeline.infer_deepseek import run_inference


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end arXiv SDG classification pipeline."
    )

    # Date range (required)
    p.add_argument("--start-date", required=True, help="Start date YYYY-MM-DD (inclusive).")
    p.add_argument("--end-date", required=True, help="End date YYYY-MM-DD (inclusive).")

    # Working directory
    p.add_argument("--work-dir", default="./run/output",
                   help="Directory where all stage outputs are written (default: ./run/output).")

    # Stage control
    p.add_argument("--skip-fetch", action="store_true", help="Skip metadata fetch stage.")
    p.add_argument("--skip-download", action="store_true", help="Skip PDF download stage.")
    p.add_argument("--skip-infer", action="store_true", help="Skip DeepSeek inference stage.")

    # Fetch options
    p.add_argument("--category", default="cs.RO", help="arXiv category (default: cs.RO).")
    p.add_argument("--max-results", type=int, default=50000,
                   help="Max metadata entries to fetch (default: 50000).")

    # Download options
    p.add_argument("--download-workers", type=int, default=16,
                   help="PDF download threads (default: 16).")

    # Inference options
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V3.2-Exp",
                   help="HF model ID (default: deepseek-ai/DeepSeek-V3.2-Exp).")
    p.add_argument("--infer-workers", type=int, default=8,
                   help="Inference threads (default: 8).")
    p.add_argument("--num-papers", type=int, default=0,
                   help="Limit inference to first N papers (0 = all).")
    p.add_argument("--max-text-chars", type=int, default=100_000,
                   help="Max PDF chars sent to model (default: 100000).")
    p.add_argument("--max-tokens", type=int, default=3000,
                   help="Max response tokens (default: 3000).")
    p.add_argument("--temperature", type=float, default=0.2,
                   help="Sampling temperature (default: 0.2).")
    p.add_argument("--force-rerun", action="store_true",
                   help="Re-run inference even for already-complete papers.")
    p.add_argument("--hf-token", default=None,
                   help="HuggingFace token (or set HF_TOKEN env var).")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    work = os.path.abspath(args.work_dir)
    os.makedirs(work, exist_ok=True)

    metadata_raw   = os.path.join(work, "metadata.json")
    pdf_folder     = os.path.join(work, "pdfs")
    metadata_local = os.path.join(work, "metadata_for_pdfs.json")
    results_json   = os.path.join(work, "results.json")

    print(f"\n{'='*60}")
    print(f"  arXiv SDG Classification Pipeline")
    print(f"  Date range : {args.start_date} → {args.end_date}")
    print(f"  Work dir   : {work}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Stage 1 — Fetch metadata
    # ------------------------------------------------------------------
    if not args.skip_fetch:
        print("── Stage 1: Fetch metadata ──────────────────────────────")
        fetch_metadata(
            start_date=args.start_date,
            end_date=args.end_date,
            output=metadata_raw,
            category=args.category,
            max_results=args.max_results,
        )
    else:
        print("── Stage 1: SKIPPED (--skip-fetch) ──────────────────────")
        if not os.path.exists(metadata_raw):
            print(f"Error: metadata file not found at '{metadata_raw}'. Cannot continue.")
            sys.exit(1)

    # ------------------------------------------------------------------
    # Stage 2 — Download PDFs
    # ------------------------------------------------------------------
    if not args.skip_download:
        print("\n── Stage 2: Download PDFs ───────────────────────────────")
        download_pdfs(
            metadata_json=metadata_raw,
            output_folder=pdf_folder,
            start_date=args.start_date,
            end_date=args.end_date,
            workers=args.download_workers,
        )
        print("\n── Stage 2b: Prepare local metadata ─────────────────────")
        prepare_metadata(
            metadata_json=metadata_raw,
            pdf_folder=pdf_folder,
            output=metadata_local,
        )
    else:
        print("── Stage 2: SKIPPED (--skip-download) ───────────────────")
        if not os.path.exists(metadata_local):
            # Fall back to raw metadata (inference will use abstract if PDF missing)
            print(f"Warning: '{metadata_local}' not found; using raw metadata.")
            metadata_local = metadata_raw

    # ------------------------------------------------------------------
    # Stage 3 — DeepSeek inference
    # ------------------------------------------------------------------
    if not args.skip_infer:
        print("\n── Stage 3: DeepSeek inference ──────────────────────────")
        run_inference(
            metadata_json=metadata_local,
            output=results_json,
            model=args.model,
            workers=args.infer_workers,
            max_text_chars=args.max_text_chars,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            num_papers=args.num_papers,
            hf_token=args.hf_token,
            force_rerun=args.force_rerun,
        )
    else:
        print("── Stage 3: SKIPPED (--skip-infer) ──────────────────────")

    print(f"\n{'='*60}")
    print("  Pipeline complete.")
    print(f"  Results : {results_json}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
