#!/usr/bin/env python3
"""
Run DeepSeek-V3.2-Exp inference on a metadata JSON of arXiv papers.

Features:
- HuggingFace InferenceClient with streaming responses
- PDF text extraction with abstract fallback
- Thread-safe incremental atomic saves after every paper
- Resume: skip papers that already have a successful response
- --force-rerun to reprocess all records

Adapted from multiple_prompt_multithread_hf.py.
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import PyPDF2
from huggingface_hub import InferenceClient

DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3.2-Exp"
DEFAULT_WORKERS = 8
DEFAULT_MAX_TEXT_CHARS = 100_000
DEFAULT_MAX_TOKENS = 3000
DEFAULT_TEMPERATURE = 0.2

_PROMPTS_DIR = Path(__file__).parent.parent / "data" / "prompts"

YES_VALUES = {"yes", "y", "true", "1"}

_write_lock = threading.Lock()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run DeepSeek inference on arXiv papers and save structured results."
    )
    p.add_argument("metadata_json", help="Path to metadata JSON (with pdf_path fields).")
    p.add_argument("--output", required=True, help="Path to write (or resume) results JSON.")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"HF model ID (default: {DEFAULT_MODEL}).")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"Parallel threads (default: {DEFAULT_WORKERS}).")
    p.add_argument("--max-text-chars", type=int, default=DEFAULT_MAX_TEXT_CHARS,
                   help=f"Max PDF chars sent to model (default: {DEFAULT_MAX_TEXT_CHARS}).")
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                   help=f"Max response tokens (default: {DEFAULT_MAX_TOKENS}).")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                   help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE}).")
    p.add_argument("--num-papers", type=int, default=0,
                   help="Process only the first N papers (0 = all).")
    p.add_argument("--hf-token", default=None,
                   help="HuggingFace token (or set HF_TOKEN env var).")
    p.add_argument("--sdg-text", default=str(_PROMPTS_DIR / "un_sdgs.txt"),
                   help="Path to UN SDGs reference text file.")
    p.add_argument("--ifr-text", default=str(_PROMPTS_DIR / "ifc_sdg_proposals.txt"),
                   help="Path to IFR proposals reference text file.")
    p.add_argument("--force-rerun", action="store_true",
                   help="Ignore existing results and re-run all papers.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Resource loading
# ---------------------------------------------------------------------------

def _load_text_file(path: str, label: str) -> str:
    if not os.path.exists(path):
        print(f"Warning: {label} file not found at '{path}'. Prompt will omit it.")
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str, max_chars: int) -> str:
    text = ""
    try:
        with open(pdf_path, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            for page in reader.pages:
                chunk = (page.extract_text() or "")
                chunk = chunk.encode("utf-8", errors="ignore").decode("utf-8")
                text += chunk + "\n"
                if len(text) >= max_chars:
                    return text[:max_chars]
    except Exception as e:
        print(f"  PDF extraction failed for '{pdf_path}': {e}")
    return text[:max_chars]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_prompt(sdg_text: str, ifr_text: str, title: str, paper_text: str):
    system = f"""
       
    My aim is to assess and quantify how often robotics research is explicitly motivated by sustainable development, to raise awareness about the prioritization of sustainability in the field. You are an expert in analyzing academic papers for their impact on the UN Sustainable Development Goals (SDGs) and related environmental or social impacts. Your task is to analyze the provided paper and identify which SDGs it most directly supports, based on its content and the IFR proposals.

    Use the full list of 17 SDGs and targets:
    {sdg_text}

    Also use the following text as a reference for the SDGs and targets. It shows the proposals of the International Federation of Robotics (IFR) on how robots can help achieve the SDGs:
    {ifr_text}

    Respond in the format:


Point 0. provide the type of the paper (survey, experimental, theoretical, report or other - or a combination of these).
If you are not sure, just say "other".
Point 1. SDGs and targets the paper is **explicitly motivated by or aims to address** (i.e., the problem or impact the authors are directly targeting). 
Point 2. provide a list of SDGs and targets SDGs and targets **relevant to the technologies or methods developed** in the paper, even if **not mentioned** or motivated by sustainability. This list should include SDGs the list in point 1.
Point 3. check if the terms "sustainability", "ecological impact", "social impact" or their derivatives are mentioned in the text, and provide a yes/no answer for each. Also check if the authors mention the UN's 17 sustainable development goals explicitly.
Point 4. **IFR Proposals:** Does the paper results/technology coincide with the International Federation of Robotics (IFR) proposals for supporting SDGs? Provide a list of the the SDGs the paper supports according to the IFR proposals, and a list of matching IFR use cases (quote or paraphrase from IFR proposals). If the paper does not match any IFR proposal, provide an empty list.
Point 5. provide a reasoning for the choices made in points 0-4, with quotes from the paper if possible, make it concise and to the point.


    """

    user = f"""
Title: {title}
Full paper text:
\"\"\"
{paper_text}
\"\"\"


Do not be verbose, keep it concise.
Do not be overly optimistic or pessimistic, just state the facts.
If not enough information is available, say so and provide with empty lists or "unknown" where appropriate.

Respond in the format:
---------------------------
0. Paper type: [survey, experimental, theoretical, report, other]
1. SDGs and targets the paper is **explicitly motivated by or aims to address** (i.e., the problem or impact the authors are directly targeting) only if they are motivated by sustainability not the technology itself:
   - SDGs: [SDG X, SDG Y, ...]
   - Targets: [[X.Y, X.Z, ...], [X.Y, ...], ...]
   - Quote(s) from the motivation/introduction.
2. SDGs and targets **relevant to the technologies or methods developed** in the paper, even if not motivated by sustainability but mentioned in the text:
   - SDGs: [SDG X, SDG Y, ...]
   - Targets: [[X.Y, X.Z, ...], [X.Y, ...], ...]
   - Brief justification for each.
3. Authors mention in the text:
   - UN SDGs: yes/no
   - Sustainability impact: yes/no
   - Ecological impact: yes/no
   - Social impact: yes/no
4. **IFR Proposals:** Does the paper results/technology coincide with the International Federation of Robotics (IFR) proposals for supporting SDGs?  
   - IFR-aligned SDGs/targets: [SDG X, SDG Y, ...]  
   - Matching IFR use cases (quote or paraphrase from IFR proposals):  
     - [E.g., “Robots used in the development & testing of drugs” or “Inspection robots enable leak detection in pipes”]  
   - Brief justification/explanation.
5. Reasoning: "A concise summary of why these SDGs and IFR alignments were chosen, quoting the paper where possible."
------------------------
""" 
    return system, user


def _normalize_response_text(text: str) -> str:
    """
    Normalize raw model text to avoid parser noise:
    - normalize newlines
    - remove wrapper separator lines
    - if multiple full answer blocks exist, keep the first one
    """
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

    # Keep from first section header onward if any leading chatter exists.
    first = re.search(r"(?im)^\s*(?:0\.|point\s*0)", t)
    if first:
        t = t[first.start():].strip()

    # If separator is followed by a new Point 0, that indicates a duplicated block.
    dup_after_sep = re.search(r"-{10,}\s*\n\s*(?:0\.|point\s*0)", t, flags=re.IGNORECASE)
    if dup_after_sep:
        t = t[:dup_after_sep.start()].strip()

    # If the response accidentally includes multiple complete blocks, keep first.
    starts = [m.start() for m in re.finditer(r"(?im)^\s*(?:0\.|point\s*0)", t)]
    if len(starts) > 1:
        t = t[:starts[1]].strip()

    return t


def _append_stream_piece(accumulated: str, piece: str) -> str:
    """
    Assemble streaming output robustly.
    Some providers emit cumulative snapshots instead of strict deltas.
    """
    if not piece:
        return accumulated
    if not accumulated:
        return piece

    # Cumulative snapshot: replace with latest full snapshot.
    if piece.startswith(accumulated):
        return piece

    # Pure duplicate trailing token/chunk.
    if accumulated.endswith(piece):
        return accumulated

    # Stitch overlapping suffix/prefix to avoid repeated seams.
    max_overlap = min(len(accumulated), len(piece))
    for k in range(max_overlap, 0, -1):
        if accumulated.endswith(piece[:k]):
            return accumulated + piece[k:]

    return accumulated + piece


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_response(text: str) -> Dict:
    parsed: Dict = {
        "paper_type": "unknown",
        "explicit_sdgs": set(),
        "relevant_sdgs": set(),
        "all_sdgs": set(),
        "mentions": {
            "un_sdgs": False,
            "sustainability": False,
            "ecological": False,
            "social": False,
        },
    }
    sdg_pattern = r"SDG\s*(\d+)"
    section = None

    for line in text.split("\n"):
        ll = line.lower()
        if "0." in line or "paper type" in ll:
            section = "type"
            for t in ("survey", "experimental", "theoretical", "report"):
                if t in ll:
                    parsed["paper_type"] = t
                    break
            else:
                if "other" in ll:
                    parsed["paper_type"] = "other"
        elif "1." in line and "explicit" in ll:
            section = "explicit"
        elif "2." in line and "relevant" in ll:
            section = "relevant"
        elif "3." in line:
            section = "mentions"

        sdgs = set(re.findall(sdg_pattern, line))
        if section == "explicit" and sdgs:
            parsed["explicit_sdgs"].update(sdgs)
        elif section == "relevant" and sdgs:
            parsed["relevant_sdgs"].update(sdgs)

        if section == "mentions":
            if "un sdg" in ll and "yes" in ll:
                parsed["mentions"]["un_sdgs"] = True
            if "sustainability" in ll and "yes" in ll:
                parsed["mentions"]["sustainability"] = True
            if "ecological" in ll and "yes" in ll:
                parsed["mentions"]["ecological"] = True
            if "social" in ll and "yes" in ll:
                parsed["mentions"]["social"] = True

    parsed["all_sdgs"] = parsed["explicit_sdgs"] | parsed["relevant_sdgs"]
    return parsed


def _parsed_to_serializable(parsed: Optional[Dict]) -> Optional[Dict]:
    if not parsed:
        return None
    key = lambda x: int(x) if x.isdigit() else 99
    return {
        "paper_type": parsed["paper_type"],
        "explicit_sdgs": sorted(parsed["explicit_sdgs"], key=key),
        "relevant_sdgs": sorted(parsed["relevant_sdgs"], key=key),
        "all_sdgs": sorted(parsed["all_sdgs"], key=key),
        "mentions": parsed["mentions"],
    }
# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _extract_section_block(section_number: int, text: str) -> str:
    if not text:
        return ""
    pattern = rf"(?:^|\n)\s*{section_number}[\.)].*?(?=\n\s*{section_number + 1}[\.)]|\Z)"
    match = re.search(pattern, text, flags=re.DOTALL)
    return match.group(0) if match else ""


def _extract_field_list(section_text: str, field_name: str) -> str:
    if not section_text:
        return ""
    pattern = rf"{field_name}\s*:\s*\[([^\]]*)\]"
    match = re.search(pattern, section_text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1) if match else ""


def _parse_sdgs_from_section(section_text: str) -> List[str]:
    sdgs_raw = _extract_field_list(section_text, "SDGs")
    if not sdgs_raw:
        return []
    normalized: List[str] = []
    seen: set = set()
    for m in re.findall(r"SDG\s*\d+", sdgs_raw, flags=re.IGNORECASE):
        num_match = re.search(r"\d+", m)
        if not num_match:
            continue
        label = f"SDG {int(num_match.group(0))}"
        if label not in seen:
            seen.add(label)
            normalized.append(label)
    return normalized


def _parse_impact_flags(text: str) -> Dict:
    t = text or ""

    def has_yes(label: str) -> bool:
        match = re.search(
            rf"{label}\s*:\s*(yes|no|true|false|y|n|0|1)", t, flags=re.IGNORECASE
        )
        return bool(match) and match.group(1).strip().lower() in YES_VALUES

    return {
        "social": has_yes(r"Social\s*impact"),
        "ecological": has_yes(r"Ecological\s*impact"),
        "sustainability": has_yes(r"Sustainability\s*impact"),
        "un_sdgs": has_yes(r"UN\s*SDGs"),
    }


def _parse_response(text: str) -> Dict:
    section0 = _extract_section_block(0, text)
    paper_type = "unknown"
    if section0:
        s0l = section0.lower()
        for t in ("survey", "experimental", "theoretical", "report", "other"):
            if t in s0l:
                paper_type = t
                break

    impacts = _parse_impact_flags(text)
    return {
        "paper_type": paper_type,
        "motivated_by_sdgs": _parse_sdgs_from_section(_extract_section_block(1, text)),
        "aligned_with_sdgs": _parse_sdgs_from_section(_extract_section_block(2, text)),
        "mentions_social_impact": impacts["social"],
        "mentions_ecological_impact": impacts["ecological"],
        "mentions_sustainability_impact": impacts["sustainability"],
        "mentions_un_sdgs": impacts["un_sdgs"],
    }


def _parsed_to_serializable(parsed: Optional[Dict]) -> Optional[Dict]:
    return parsed if parsed else None


# ---------------------------------------------------------------------------
# Inference call (single paper)
# ---------------------------------------------------------------------------

def _call_model(
    client: InferenceClient,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
) -> Dict:
    t0 = time.time()
    response_text = ""
    try:
        for chunk in client.chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        ):
            if chunk.choices and chunk.choices[0].delta.content:
                response_text = _append_stream_piece(response_text, chunk.choices[0].delta.content)

        response_text = _normalize_response_text(response_text)
        elapsed = time.time() - t0
        parsed = _parse_response(response_text)
        return {
            "success": True,
            "response": response_text,
            "parsed": _parsed_to_serializable(parsed),
            "model": model,
            "time_seconds": round(elapsed, 2),
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "response": None,
            "parsed": None,
            "model": model,
            "time_seconds": round(time.time() - t0, 2),
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _is_complete(paper: Dict) -> bool:
    d = paper.get("deepseek") or {}
    return bool(d.get("success")) and bool(d.get("response"))


def _merge_existing(papers: List[Dict], output: str) -> int:
    if not os.path.exists(output):
        return 0
    try:
        with open(output, "r", encoding="utf-8") as f:
            existing = json.load(f)
    except Exception as e:
        print(f"Warning: could not load '{output}': {e}")
        return 0
    idx = {p.get("id"): p for p in existing if p.get("id")}
    merged = 0
    for paper in papers:
        pid = paper.get("id")
        if pid in idx:
            prev = idx[pid]
            if "deepseek" in prev:
                paper["deepseek"] = prev["deepseek"]
            merged += 1
    return merged


# ---------------------------------------------------------------------------
# Atomic save
# ---------------------------------------------------------------------------

def _save(output: str, papers: List[Dict]) -> None:
    tmp = output + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    os.replace(tmp, output)


# ---------------------------------------------------------------------------
# Per-paper worker
# ---------------------------------------------------------------------------

def _annotate_paper(
    paper: Dict,
    idx: int,
    total: int,
    sdg_text: str,
    ifr_text: str,
    client: InferenceClient,
    model: str,
    output: str,
    all_papers: List[Dict],
    max_text_chars: int,
    max_tokens: int,
    temperature: float,
    force_rerun: bool,
) -> Dict:
    title = (paper.get("title") or "").strip()
    abstract = (paper.get("summary") or "").strip()
    pdf_path = paper.get("pdf_path") or ""

    if not title:
        print(f"[{idx+1}/{total}] Skipping — missing title.")
        return paper

    if not force_rerun and _is_complete(paper):
        print(f"[{idx+1}/{total}] Already done: {title[:80]}")
        return paper

    print(f"[{idx+1}/{total}] Processing: {title[:80]}")

    # Extract text
    paper_text = ""
    if pdf_path and os.path.exists(pdf_path):
        paper_text = extract_text_from_pdf(pdf_path, max_text_chars)
    if not paper_text:
        print(f"  No PDF text — falling back to abstract.")
        paper_text = abstract

    system, user = _build_prompt(sdg_text, ifr_text, title, paper_text)
    result = _call_model(client, model, system, user, max_tokens, temperature)
    paper["deepseek"] = result

    if result["success"]:
        print(f"[{idx+1}/{total}] ✓ done in {result['time_seconds']}s")
    else:
        print(f"[{idx+1}/{total}] ✗ error: {result['error']}")

    with _write_lock:
        _save(output, all_papers)

    return paper


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_inference(
    metadata_json: str,
    output: str,
    model: str = DEFAULT_MODEL,
    workers: int = DEFAULT_WORKERS,
    max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    num_papers: int = 0,
    hf_token: Optional[str] = None,
    sdg_text_path: str = str(_PROMPTS_DIR / "un_sdgs.txt"),
    ifr_text_path: str = str(_PROMPTS_DIR / "ifc_sdg_proposals.txt"),
    force_rerun: bool = False,
) -> List[Dict]:
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("Error: no HuggingFace token. Set HF_TOKEN or pass --hf-token.")
        sys.exit(1)

    sdg_text = _load_text_file(sdg_text_path, "UN SDGs")
    ifr_text = _load_text_file(ifr_text_path, "IFR proposals")

    with open(metadata_json, "r", encoding="utf-8") as f:
        papers = json.load(f)

    if num_papers > 0:
        papers = papers[:num_papers]

    merged = _merge_existing(papers, output)
    if merged:
        print(f"Resumed: loaded existing results for {merged}/{len(papers)} papers.")

    # Initial save so the output file exists early
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    _save(output, papers)

    # Skip already-complete papers
    if not force_rerun:
        pending = [(i, p) for i, p in enumerate(papers) if not _is_complete(p)]
        done_count = len(papers) - len(pending)
        if done_count:
            print(f"Skipping {done_count} already-complete papers; {len(pending)} remaining.")
    else:
        pending = list(enumerate(papers))

    if not pending:
        print("Nothing to do — all papers already complete.")
        return papers

    print(f"Running inference with model '{model}' on {len(pending)} papers, {workers} threads ...")
    client = InferenceClient(token=token)
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _annotate_paper,
                paper, idx, len(papers),
                sdg_text, ifr_text,
                client, model, output, papers,
                max_text_chars, max_tokens, temperature, force_rerun,
            ): idx
            for idx, paper in pending
        }
        completed = 0
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 10 == 0 or completed == len(futures):
                print(f"Progress: {completed}/{len(futures)} processed")

    print(f"Done in {time.time() - t0:.1f}s. Output: '{output}'")
    return papers


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    run_inference(
        metadata_json=args.metadata_json,
        output=args.output,
        model=args.model,
        workers=args.workers,
        max_text_chars=args.max_text_chars,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        num_papers=args.num_papers,
        hf_token=args.hf_token,
        sdg_text_path=args.sdg_text,
        ifr_text_path=args.ifr_text,
        force_rerun=args.force_rerun,
    )


if __name__ == "__main__":
    main()
