"""
TALMAS hyperparameter sweep — one config ID per run, designed for distribution
across multiple machines.

List all configs:
  python scripts/run_sweep.py --list-configs

Run a single config (one machine):
  python scripts/run_sweep.py --config-id 12 \\
      --model GSAI-ML/LLaDA-8B-Instruct \\
      --max-samples 100 --steps 256 \\
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import SWEEP_CONFIGS, SWEEP_CONFIG_BY_ID
from scripts.gsm8k_eval import evaluate


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
    os.makedirs(args.output_dir, exist_ok=True)

    ckpt     = os.path.join(args.output_dir, f"sweep_cfg{args.config_id:02d}.jsonl")
    csv_path = os.path.join(args.output_dir, "sweep_results.csv")

    print(f"Config {args.config_id}: λ_max={cfg_meta['lambda_max']}  "
          f"μ={cfg_meta['mu']}  slope={cfg_meta['sigmoid_slope']}")

    # Build the Namespace that gsm8k_eval.evaluate() expects
    eval_args = argparse.Namespace(
        model=args.model,
        split=args.split,
        max_samples=args.max_samples,
        generation_length=args.generation_length,
        steps=args.steps,
        # TALMAS — always enabled for sweep configs
        talmas=True,
        lambda_max=cfg_meta["lambda_max"],
        mu=cfg_meta["mu"],
        sigmoid_slope=cfg_meta["sigmoid_slope"],
        no_timestep_gate=not cfg_meta.get("use_timestep_gate", True),
        no_layer_gate=not cfg_meta.get("use_layer_gate", True),
        # Checkpointing — written by gsm8k_eval; we read it back for the CSV
        checkpoint=ckpt,
        verbose=False,
        # Output — we write the CSV ourselves; suppress gsm8k_eval's JSON save
        output_dir=None,
        output_file=None,
    )

    accuracy_pct = evaluate(eval_args)  # returns accuracy as a percentage

    # Reconstruct correct/total from the checkpoint file gsm8k_eval wrote
    correct = total = 0
    if os.path.exists(ckpt):
        with open(ckpt) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    correct += int(entry.get("correct", False))
                    total   += 1

    _append_csv(csv_path, {
        "config_id":     args.config_id,
        "tier":          cfg_meta["tier"],
        "lambda_max":    cfg_meta["lambda_max"],
        "mu":            cfg_meta["mu"],
        "sigmoid_slope": cfg_meta["sigmoid_slope"],
        "correct":       correct,
        "total":         total,
        "accuracy":      round(accuracy_pct / 100, 6),
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
    parser.add_argument("--model",  type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--split",  type=str, default="test", choices=["train", "test"])
    parser.add_argument("--max-samples",       type=int,   default=100,  dest="max_samples")
    parser.add_argument("--steps",             type=int,   default=256)
    parser.add_argument("--generation-length", type=int,   default=256, dest="generation_length")
    parser.add_argument("--output-dir",        type=str,   default="results/sweep", dest="output_dir")
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
        print(f"Error: unknown config_id {args.config_id}. "
              f"Valid: {sorted(SWEEP_CONFIG_BY_ID.keys())}")
        sys.exit(1)

    run(args)
