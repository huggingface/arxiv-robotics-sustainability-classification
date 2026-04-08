#!/usr/bin/env python3
"""
Validate the DeepSeek inference output JSON and report completion stats.

Exit code:
  0  — all papers complete
  1  — some papers pending or failed (safe to resume)
"""

import argparse
import json
import os
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate DeepSeek inference output JSON.")
    p.add_argument("output_json", help="Path to the inference results JSON.")
    return p.parse_args()


def validate(output_json: str) -> bool:
    """
    Print a completion report. Returns True if all papers are complete.
    """
    if not os.path.exists(output_json):
        print(f"File not found: {output_json}")
        return False

    with open(output_json, "r", encoding="utf-8") as f:
        papers = json.load(f)

    total = len(papers)
    if total == 0:
        print("Output file is empty.")
        return False

    success = sum(1 for p in papers if p.get("deepseek", {}).get("success") is True)
    failed = sum(1 for p in papers if p.get("deepseek", {}).get("success") is False)
    pending = total - success - failed

    print("\n=== Validation Report ===")
    print(f"File:       {output_json}")
    print(f"Total:      {total}")
    print(f"  Complete: {success}  ({100 * success / total:.1f}%)")
    print(f"  Failed:   {failed}  ({100 * failed / total:.1f}%)")
    print(f"  Pending:  {pending}  ({100 * pending / total:.1f}%)")

    if failed:
        print("\nFailed papers:")
        for p in papers:
            d = p.get("deepseek") or {}
            if d.get("success") is False:
                print(
                    f"  [{p.get('id', '?')}] "
                    f"{(p.get('title') or '')[:70]} "
                    f"— {d.get('error', '?')}"
                )

    resumable = pending + failed > 0
    print(f"\nResumable: {'YES — re-run without --force-rerun to continue' if resumable else 'NO — all done'}")
    return not resumable


def main() -> None:
    args = parse_args()
    all_done = validate(args.output_json)
    sys.exit(0 if all_done else 1)


if __name__ == "__main__":
    main()
