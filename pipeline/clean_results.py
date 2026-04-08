#!/usr/bin/env python3
"""
Clean duplicated/concatenated DeepSeek responses in an existing results JSON file.

Usage:
  python -m pipeline.clean_results run/2026-april/results.json
"""

import argparse
import json
import os
import re


def _normalize_response_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    t = text.replace("\r\n", "\n").strip()
    if not t:
        return ""

    sep = "---------------------------"
    lines = t.split("\n")
    while lines and lines[0].strip() == sep:
        lines.pop(0)
    while lines and lines[-1].strip() == sep:
        lines.pop()
    t = "\n".join(lines).strip()

    first = re.search(r"(?im)^\s*(?:0\.|point\s*0)", t)
    if first:
        t = t[first.start():].strip()

    dup_after_sep = re.search(r"-{10,}\s*\n\s*(?:0\.|point\s*0)", t, flags=re.IGNORECASE)
    if dup_after_sep:
        t = t[:dup_after_sep.start()].strip()

    starts = [m.start() for m in re.finditer(r"(?im)^\s*(?:0\.|point\s*0)", t)]
    if len(starts) > 1:
        t = t[:starts[1]].strip()

    return t


def clean_results(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    changed = 0
    for paper in data:
        deepseek = paper.get("deepseek") or {}
        resp = deepseek.get("response")
        if not isinstance(resp, str):
            continue
        cleaned = _normalize_response_text(resp)
        if cleaned != resp:
            deepseek["response"] = cleaned
            paper["deepseek"] = deepseek
            changed += 1

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

    return changed


def main() -> None:
    p = argparse.ArgumentParser(description="Clean duplicated response blocks in results JSON.")
    p.add_argument("results_json", help="Path to results JSON.")
    args = p.parse_args()

    changed = clean_results(args.results_json)
    print(f"Cleaned responses in {changed} records.")


if __name__ == "__main__":
    main()
