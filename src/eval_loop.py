"""
Shared evaluation kernel used by sweep runners and other callers.
"""

import json
from typing import Optional

import torch
from tqdm import tqdm

from src.config import SamplingConfig, TALMASConfig
from src.datasets import DatasetAdapter
from src.sampling import low_confidence_remasking_sample
from src.talmas import TALMASHookManager


def eval_dataset_config(
    *,
    model,
    tokenizer,
    device: torch.device,
    mask_token_id: int,
    eos_token_id: int,
    sampling_cfg: SamplingConfig,
    talmas_cfg: TALMASConfig,
    is_instruct: bool,
    dataset,
    adapter: DatasetAdapter,
    checkpoint_path: Optional[str] = None,
    desc: str = "eval",
) -> dict:
    """
    Evaluate one TALMASConfig on a dataset slice.

    Appends each result to checkpoint_path (JSONL) immediately so a preempted
    run can resume by counting lines.  Pass checkpoint_path=None to skip.

    Returns {correct, total, accuracy, results}.
    """
    hook_manager: Optional[TALMASHookManager] = None
    if talmas_cfg.lambda_max > 0.0:
        hook_manager = TALMASHookManager(model, talmas_cfg)

    correct = 0
    total = 0
    results = []

    try:
        for example in tqdm(dataset, desc=desc):
            question = adapter.get_question(example)
            gold_ans = adapter.extract_gold(example)

            prompt_text = adapter.build_prompt(question, is_instruct)
            prompt_ids = tokenizer(
                prompt_text, return_tensors="pt", add_special_tokens=True
            ).input_ids.to(device)

            output_ids = low_confidence_remasking_sample(
                model=model,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                cfg=sampling_cfg,
                device=device,
                mask_token_id=mask_token_id,
                eos_token_id=eos_token_id,
                hook_manager=hook_manager,
            )

            output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            pred_ans = adapter.extract_answer(output_text)
            is_correct = adapter.answers_match(pred_ans, gold_ans)

            correct += int(is_correct)
            total += 1

            entry = {
                "question": question,
                "gold": gold_ans,
                "prediction": pred_ans,
                "output": output_text,
                "correct": is_correct,
            }
            results.append(entry)

            if checkpoint_path is not None:
                with open(checkpoint_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")

            status = "✓" if is_correct else "✗"
            tqdm.write(
                f"[{total:>4}] {status}  gold={str(gold_ans):<8}  pred={str(pred_ans):<8}  "
                f"running acc: {correct}/{total} ({correct / total * 100:.1f}%)"
            )
    finally:
        if hook_manager is not None:
            hook_manager.remove()

    accuracy = correct / total if total > 0 else 0.0
    return {"correct": correct, "total": total, "accuracy": accuracy, "results": results}


# Backward-compatible alias
eval_gsm8k_config = eval_dataset_config
