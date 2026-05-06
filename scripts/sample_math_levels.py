"""
Sample 50 problems from each difficulty level (Level 1–3, Algebra only) in the
MATH dataset, producing a fixed 150-question index set for TALMAS evaluation.

Filters applied before sampling:
  - type == 'Algebra'
  - level in {'Level 1', 'Level 2', 'Level 3'}
  - no [asy] in problem  (skips asymptote geometry diagrams)
  - \\boxed present in solution  (answer is parseable)

Usage:
    python scripts/sample_math_levels.py                  # save JSON + print indices
    python scripts/sample_math_levels.py --seed 42        # different seed
    python scripts/sample_math_levels.py --per-level 50   # change sample size

Output:
    results/math_level_indices.json  — saved indices with per-level breakdown
    Printed: comma-separated index string ready for --indices flag
"""

import sys
import os
import json
import random
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets import load_dataset

LEVELS = ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]


def is_valid(ex: dict) -> bool:
    return (
        ex["type"] == "Algebra"
        and ex["level"] in LEVELS
        and "[asy]" not in ex["problem"]
        and "\\boxed" in ex["solution"]
    )


def sample_indices(per_level: int = 50, seed: int = 0) -> dict:
    print("Loading MATH dataset...")
    ds = load_dataset("qwedsacf/competition_math", split="train")

    # Group dataset indices by level, applying all filters
    by_level: dict = defaultdict(list)
    for i, ex in enumerate(ds):
        if is_valid(ex):
            by_level[ex["level"]].append(i)

    print(f"Candidates after filtering (Algebra, Level 1–3, no asy, has \\boxed):")
    for lv in LEVELS:
        print(f"  {lv}: {len(by_level[lv])} problems")

    rng = random.Random(seed)
    sampled: dict = {}
    all_indices: list = []

    for lv in LEVELS:
        pool = by_level[lv]
        if len(pool) < per_level:
            print(f"  WARNING: {lv} has only {len(pool)} candidates, using all of them")
        chosen = rng.sample(pool, min(per_level, len(pool)))
        chosen.sort()
        sampled[lv] = chosen
        all_indices.extend(chosen)

    all_indices.sort()

    print(f"\nSampled {len(all_indices)} total ({per_level} per level, seed={seed})")
    for lv in LEVELS:
        print(f"  {lv}: {len(sampled[lv])} indices")

    return {"per_level": per_level, "seed": seed, "by_level": sampled, "all_indices": all_indices}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-level", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="results/math_level_indices.json")
    args = parser.parse_args()

    result = sample_indices(per_level=args.per_level, seed=args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.output}")

    indices_str = ",".join(str(i) for i in result["all_indices"])
    print(f"\n--- Copy this for --indices ---")
    print(indices_str)


if __name__ == "__main__":
    main()
