"""
TALMAS hyperparameter sweep — one config ID per run, designed for distribution
across multiple machines.

List all configs:
  python scripts/run_sweep.py --list-configs

Run a single config (one machine):
  python scripts/run_sweep.py --config-id 12 \\
      --model GSAI-ML/LLaDA-8B-Instruct \\
      --max-samples 100 --steps 128 \\
      --output-dir results/sweep

Distribute across machines (example: 5 VMs, one λ row each):
  # VM 1  (ids 1-5,  λ=1.0)
  for id in 1 2 3 4 5; do
      python scripts/run_sweep.py --config-id $id --output-dir gs_mount/sweep ...
  done

After all Tier 1 runs complete, update SWEEP_CONFIGS ids 26-28 in src/config.py
with the best (lambda_max, mu) found, then run --config-id 26/27/28 for Tier 3.
"""

import sys
import os
import csv
import json
import argparse
from datetime import datetime
from typing import Optional

import torch
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import (
    SamplingConfig, TALMASConfig,
    BASE_CONFIG, INSTRUCT_CONFIG,
    SWEEP_CONFIGS, SWEEP_CONFIG_BY_ID,
)
from src.utils import load_model_and_tokenizer, resolve_special_tokens
from src.eval_loop import eval_gsm8k_config


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "config_id", "tier", "lambda_max", "mu", "sigmoid_slope",
    "correct", "total", "accuracy", "checkpoint",
]


def _append_csv(csv_path: str, row: dict) -> None:
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in _CSV_FIELDS})


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _ckpt_path(output_dir: str, config_id: int) -> str:
    return os.path.join(output_dir, f"sweep_cfg{config_id:02d}.jsonl")


def _count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def _load_correct(path: str) -> int:
    correct = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                correct += int(json.loads(line).get("correct", False))
    return correct


# ---------------------------------------------------------------------------
# List configs
# ---------------------------------------------------------------------------

def list_configs() -> None:
    for tier_id, label in [
        (1, "Tier 1 — λ_max × μ joint grid (Full TALMAS, both gates on)"),
        (3, "Tier 3 — sigmoid slope sweep (update lambda_max/mu to best Tier 1 values first)"),
    ]:
        entries = [c for c in SWEEP_CONFIGS if c["tier"] == tier_id]
        print(f"\n{label}")
        print(f"  {'id':>3}  {'λ_max':>6}  {'μ':>5}  {'slope':>5}")
        for c in entries:
            print(f"  {c['id']:>3}  {c['lambda_max']:>6.4g}  {c['mu']:>5.4g}  "
                  f"{c['sigmoid_slope']:>5.4g}")
    print(f"\nTotal: {len(SWEEP_CONFIGS)} configs")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args) -> None:
    cfg_meta = SWEEP_CONFIG_BY_ID[args.config_id]

    is_instruct = "instruct" in args.model.lower()
    sampling_cfg = INSTRUCT_CONFIG if is_instruct else BASE_CONFIG
    if args.steps:
        sampling_cfg.steps = args.steps
    if args.generation_length:
        sampling_cfg.generation_length = args.generation_length

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt = _ckpt_path(args.output_dir, args.config_id)
    csv_path = os.path.join(args.output_dir, "sweep_results.csv")

    print(f"Config {args.config_id}: λ_max={cfg_meta['lambda_max']}  "
          f"μ={cfg_meta['mu']}  slope={cfg_meta['sigmoid_slope']}")
    print(f"Model:      {args.model}")
    print(f"Steps:      {sampling_cfg.steps}   GenLen: {sampling_cfg.generation_length}")
    print(f"Checkpoint: {ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:     {device}\n")

    # Load dataset
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split=args.split)
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    n_total = len(dataset)

    # Resume from checkpoint if partial
    offset = _count_lines(ckpt)
    prior_correct = 0
    if offset >= n_total:
        print(f"Already complete ({offset}/{n_total}). Loading from checkpoint.")
        prior_correct = _load_correct(ckpt)
        accuracy = prior_correct / offset if offset > 0 else 0.0
        print(f"Accuracy: {prior_correct}/{offset} = {accuracy:.3f}")
        return
    if offset > 0:
        prior_correct = _load_correct(ckpt)
        print(f"Resuming from {offset}/{n_total} (prior correct: {prior_correct})")
        dataset = dataset.select(range(offset, n_total))
    else:
        print(f"Evaluating {n_total} examples...\n")

    # Load model (always eager — TALMAS hook requires it)
    tokenizer, model = load_model_and_tokenizer(args.model, eager_attn=True)
    mask_token_id, eos_token_id = resolve_special_tokens(tokenizer, model)
    print(f"mask_token_id={mask_token_id}  eos_token_id={eos_token_id}\n")

    talmas_cfg = TALMASConfig(
        lambda_max=cfg_meta["lambda_max"],
        mu=cfg_meta["mu"],
        use_timestep_gate=cfg_meta.get("use_timestep_gate", True),
        use_layer_gate=cfg_meta.get("use_layer_gate", True),
        sigmoid_slope=cfg_meta["sigmoid_slope"],
        timestep_exponent=2.0,
    )

    summary = eval_gsm8k_config(
        model=model, tokenizer=tokenizer, device=device,
        mask_token_id=mask_token_id, eos_token_id=eos_token_id,
        sampling_cfg=sampling_cfg, talmas_cfg=talmas_cfg,
        is_instruct=is_instruct, dataset=dataset,
        checkpoint_path=ckpt,
        desc=f"cfg{args.config_id}",
    )

    total_correct = summary["correct"] + prior_correct
    total_done = summary["total"] + offset
    accuracy = total_correct / total_done if total_done > 0 else 0.0

    print(f"\n{'='*50}")
    print(f"Config {args.config_id} — λ={cfg_meta['lambda_max']} μ={cfg_meta['mu']} "
          f"slope={cfg_meta['sigmoid_slope']}")
    print(f"Accuracy: {total_correct}/{total_done} = {accuracy:.3f}")
    print(f"{'='*50}")

    _append_csv(csv_path, {
        "config_id":     args.config_id,
        "tier":          cfg_meta["tier"],
        "lambda_max":    cfg_meta["lambda_max"],
        "mu":            cfg_meta["mu"],
        "sigmoid_slope": cfg_meta["sigmoid_slope"],
        "correct":       total_correct,
        "total":         total_done,
        "accuracy":      round(accuracy, 6),
        "checkpoint":    ckpt,
    })
    print(f"Row appended to {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TALMAS sweep — run one config ID per machine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--list-configs", action="store_true",
        help="Print all available sweep configs and exit",
    )
    parser.add_argument(
        "--config-id", type=int, default=None, dest="config_id",
        help="Sweep config ID to run (see --list-configs)",
    )
    parser.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--max-samples", type=int, default=100, dest="max_samples")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument(
        "--generation-length", type=int, default=None, dest="generation_length",
    )
    parser.add_argument("--output-dir", type=str, default="results/sweep", dest="output_dir")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    if args.list_configs:
        list_configs()
        sys.exit(0)

    if args.config_id is None:
        print("Error: provide --config-id N or --list-configs")
        build_parser().print_help()
        sys.exit(1)

    if args.config_id not in SWEEP_CONFIG_BY_ID:
        valid = sorted(SWEEP_CONFIG_BY_ID.keys())
        print(f"Error: config_id {args.config_id} not found. Valid ids: {valid}")
        sys.exit(1)

    run(args)
