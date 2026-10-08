#!/usr/bin/env python3
"""
intent_cli.py — Interactive CLINC150 Intent Classification CLI

Loads a fine-tuned sentence-transformers/all-MiniLM-L12-v2 sequence
classification model and runs interactive intent prediction with
confidence scoring, OOS fallback detection, and rich terminal output.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ──────────────────────────── ANSI helpers ────────────────────────────────────
class Style:
    """Minimal ANSI colour/style codes (no external dependencies)."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    CYAN    = "\033[96m"
    RED     = "\033[91m"
    MAGENTA = "\033[95m"
    WHITE   = "\033[97m"
    GREY    = "\033[90m"
    UNDERLINE = "\033[4m"

    # Backgrounds
    BG_GREEN  = "\033[42m"
    BG_YELLOW = "\033[43m"


def styled(text: str, *codes: str) -> str:
    return "".join(codes) + str(text) + Style.RESET


# ──────────────────────────── Model loader ────────────────────────────────────
def load_model(model_path: str = "Kami0867/MiniLM-L12-Intent-Matching", revision: Optional[str] = None):
    """Load tokenizer & model, resolve device, and extract id2label."""
    print(styled("[*] Loading model and tokenizer ...", Style.DIM))

    pretrained_kwargs = {}
    if revision:
        pretrained_kwargs["revision"] = revision

    tokenizer = AutoTokenizer.from_pretrained(model_path, **pretrained_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, **pretrained_kwargs)
    calibration_path = Path(model_path) / "confidence_calibration.json"
    if calibration_path.is_file():
        with calibration_path.open(encoding="utf-8") as calibration_file:
            calibration = json.load(calibration_file)
    else:
        # Load calibrated parameters from Hugging Face Hub if model_path is a repository ID
        try:
            from huggingface_hub import hf_hub_download
            calib_file = hf_hub_download(
                repo_id=model_path,
                filename="confidence_calibration.json",
                **pretrained_kwargs,
            )
            with open(calib_file, "r", encoding="utf-8") as calibration_file:
                calibration = json.load(calibration_file)
        except Exception:
            calibration = {"temperature": 1.0, "oos_threshold": 0.0, "max_length": 64}

    # Device selection: CUDA → CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_label = f"CUDA ({torch.cuda.get_device_name(0)})"
    else:
        device = torch.device("cpu")
        device_label = "CPU"

    model.to(device)
    model.eval()

    # id2label from config (set during fine-tuning)
    id2label = model.config.id2label
    if not id2label:
        # Fallback: attempt to load labels from the HF dataset
        try:
            from datasets import load_dataset
            ds = load_dataset("clinc_oos", "plus", split="train", trust_remote_code=True)
            labels = sorted(set(ds["intent"]))
            id2label = {i: lbl for i, lbl in enumerate(labels)}
            print(styled("[!] id2label loaded from HF datasets fallback.", Style.YELLOW))
        except Exception:
            print(styled("[X] Could not resolve id2label mapping. Exiting.", Style.RED))
            sys.exit(1)

    num_labels = len(id2label)

    print(styled(f"[OK] Model loaded  ", Style.GREEN, Style.BOLD)
          + styled(f"({num_labels} classes, device: {device_label})", Style.DIM))
    print(styled(
        f"[i] Confidence temperature={calibration['temperature']:.3f}, "
        f"OOS threshold={calibration['oos_threshold']:.4f}",
        Style.DIM,
    ))
    return tokenizer, model, device, id2label, calibration


# ──────────────────────────── Inference ────────────────────────────────────────
def predict(text: str, tokenizer, model, device, id2label, calibration, threshold=None):
    """Run inference and return formatted results."""
    temperature = float(calibration.get("temperature", 1.0))
    max_length = int(calibration.get("max_length", 64))
    threshold = float(
        calibration.get("oos_threshold", 0.0) if threshold is None else threshold
    )
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = F.softmax(logits / max(temperature, 1e-6), dim=-1).squeeze(0)
    top_k = torch.topk(probs, k=min(4, len(probs)))

    top_id    = top_k.indices[0].item()
    top_score = top_k.values[0].item()
    raw_top_label = id2label.get(top_id, id2label.get(str(top_id), f"LABEL_{top_id}"))

    rejected_as_oos = top_label_is_oos = raw_top_label == "oos"
    if top_score < threshold and raw_top_label != "oos":
        rejected_as_oos = True
    top_label = "oos" if rejected_as_oos else raw_top_label
    # ── Header: predicted intent ──────────────────────────────────────────
    conf_pct = f"{top_score * 100:.1f}%"
    if rejected_as_oos:
        conf_colour = Style.YELLOW
        badge = styled(" OUT OF SCOPE ", Style.BOLD, Style.BG_YELLOW, Style.WHITE)
    else:
        conf_colour = Style.GREEN
        badge = ""

    print()
    print(styled("  +---------------------------------------------------+", Style.DIM))
    intent_display = styled(top_label, Style.CYAN, Style.BOLD)
    print(f"  |  >>  Predicted Intent:  {intent_display}")
    print(f"  |  %%  Confidence:        {styled(conf_pct, conf_colour, Style.BOLD)}  {badge}")

    if rejected_as_oos:
        if top_label_is_oos:
            message = "Model classified this request as out-of-scope."
        else:
            message = f"Rejected as out-of-scope; strongest intent match was {raw_top_label}."
        print(f"  |  {styled('[!] ' + message, Style.YELLOW)}")

    print(styled("  +---------------------------------------------------+", Style.DIM))

    # ── Top alternatives ──────────────────────────────────────────────────
    alt_indices = top_k.indices[1:4]
    alt_scores  = top_k.values[1:4]

    print()
    print(styled("  Top alternatives:", Style.BOLD))
    print(styled("  -----------------------------------------", Style.DIM))
    for rank, (idx, score) in enumerate(zip(alt_indices, alt_scores), start=2):
        label = id2label.get(idx.item(), id2label.get(str(idx.item()), f"LABEL_{idx.item()}"))
        pct = f"{score.item() * 100:.1f}%"
        bar_len = int(score.item() * 30)
        bar = "#" * bar_len + "." * (30 - bar_len)
        print(f"  {styled(f'#{rank}', Style.DIM)}  {styled(label, Style.CYAN):<40s} "
              f"{styled(bar, Style.DIM)}  {styled(pct, Style.WHITE)}")

    print()


# ──────────────────────────── CLI loop ─────────────────────────────────────────
def banner():
    print()
    print(styled("+======================================================+", Style.CYAN))
    print(styled("|", Style.CYAN)
          + styled("  CLINC150 Intent Classifier  ", Style.BOLD, Style.WHITE)
          + styled("*", Style.DIM) + styled(" MiniLM-L12-v2 ", Style.DIM)
          + styled("  |", Style.CYAN))
    print(styled("+======================================================+", Style.CYAN))
    print(styled("  Type a sentence to classify its intent.", Style.DIM))
    print(styled("  Commands: ", Style.DIM) + styled("exit", Style.YELLOW)
          + styled(" | ", Style.DIM) + styled("quit", Style.YELLOW)
          + styled(" | ", Style.DIM) + styled("Ctrl+C", Style.YELLOW))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive CLINC150 intent classification CLI."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Kami0867/MiniLM-L12-Intent-Matching",
        help="Path to the fine-tuned model directory or Hugging Face Hub repository "
             "(default: Kami0867/MiniLM-L12-Intent-Matching)",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional model revision or commit hash on Hugging Face Hub "
             "(e.g., 11d37c733b9bed728da33405400591b408fe59c1).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the validation-calibrated confidence threshold for OOS rejection.",
    )
    args = parser.parse_args()

    tokenizer, model, device, id2label, calibration = load_model(args.model_path, args.revision)
    banner()

    while True:
        try:
            user_input = input(styled("User > ", Style.BOLD, Style.MAGENTA))
        except (KeyboardInterrupt, EOFError):
            print(styled("\n[~] Goodbye!", Style.CYAN, Style.BOLD))
            break

        text = user_input.strip()

        if not text:
            continue
        if text.lower() in {"exit", "quit"}:
            print(styled("[~] Goodbye!", Style.CYAN, Style.BOLD))
            break

        predict(text, tokenizer, model, device, id2label, calibration, args.threshold)


if __name__ == "__main__":
    main()
