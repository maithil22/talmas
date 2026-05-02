"""
LLaDA reverse-diffusion sampling  (Algorithm 5: low-confidence remasking).

Extracted verbatim from llada_gsm8k_eval.py with one addition: an optional
hook_manager parameter that updates TALMAS state before each forward pass.
All Algorithm 5 logic is unchanged.
"""

import math
from typing import Optional

import torch
import logging
import torch.nn.functional as F

from src.config import SamplingConfig
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@torch.inference_mode()
def low_confidence_remasking_sample(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,          # (1, prompt_len)  — already on device
    cfg: SamplingConfig,
    device: torch.device,
    mask_token_id: int,
    eos_token_id: int,
    hook_manager=None,                 # TALMASHookManager | None
    diagnostics=None,                  # DiagnosticsCollector | None
) -> torch.Tensor:
    """
    Pure diffusion sampling with low-confidence remasking (Algorithm 5).

    If hook_manager is provided, its state is updated at each diffusion step
    so that TALMAS logit biases are applied during the forward pass.

    If diagnostics is provided, it receives begin_step / end_step calls each
    iteration so attention weights, confidence, and suppression can be captured.

    Returns the generated token ids as a 1-D tensor (response only).
    """
    L = cfg.generation_length
    N = cfg.steps

    # ------------------------------------------------------------------ #
    # Step 1 — initialise: fully masked response appended to prompt        #
    # ------------------------------------------------------------------ #
    response = torch.full((1, L), mask_token_id, dtype=torch.long, device=device)
    input_ids = torch.cat([prompt_ids, response], dim=1)  # (1, prompt_len + L)
    prompt_len = prompt_ids.shape[1]

    # uniform time-steps: t goes from 1 → 1/N in steps of 1/N
    timesteps = torch.linspace(1.0, 1.0 / N, N, device=device)

    for i, t in enumerate(timesteps):
        s = t - 1.0 / N          # next time-step (can be 0 at last step)
        s = max(s.item(), 0.0)
        t_val = t.item()

        # -------------------------------------------------------------- #
        # TALMAS + diagnostics: compute mask positions once, shared by both#
        # -------------------------------------------------------------- #
        needs_mask = hook_manager is not None or diagnostics is not None
        mask_positions = (input_ids == mask_token_id) if needs_mask else None  # (1, S)

        if hook_manager is not None:
            hook_manager.set_state(r_t=t_val, mask_positions=mask_positions)

        if diagnostics is not None:
            diagnostics.begin_step(i, t_val, mask_positions)

        # -------------------------------------------------------------- #
        # Step 2 — forward pass: predict all masked tokens simultaneously #
        # -------------------------------------------------------------- #
        logits = model(input_ids=input_ids).logits  # (1, prompt_len+L, vocab)
        response_logits = logits[0, prompt_len:, :]  # (L, vocab)

        probs = F.softmax(response_logits, dim=-1)   # (L, vocab)

        # Greedy prediction and confidence score for each position
        pred_ids   = probs.argmax(dim=-1)            # (L,)
        confidence = probs.max(dim=-1).values        # (L,)

        # -------------------------------------------------------------- #
        # Optional: zero out EOS confidence (instruct model only)         #
        # -------------------------------------------------------------- #
        if cfg.zero_eos_confidence:
            eos_mask = (pred_ids == eos_token_id)
            confidence = confidence.masked_fill(eos_mask, 0.0)

        # -------------------------------------------------------------- #
        # Step 3 — decide which positions to keep vs remask               #
        # -------------------------------------------------------------- #
        # Positions that were already unmasked stay unmasked (confidence=1)
        current_response = input_ids[0, prompt_len:]  # (L,)
        already_unmasked  = (current_response != mask_token_id)
        confidence = confidence.masked_fill(already_unmasked, 1.0)
        pred_ids   = torch.where(already_unmasked, current_response, pred_ids)

        if diagnostics is not None:
            diagnostics.end_step(i, confidence)

        # Number of tokens that should be unmasked at time s
        n_unmask = math.floor(L * (1.0 - s))
        n_unmask = max(0, min(n_unmask, L))

        # Select the n_unmask positions with the highest confidence.
        # Tiebreak by position (earlier index wins) to make selection deterministic
        # when bfloat16 rounds two probabilities to the same value.  The penalty
        # (1e-7 * position) is far below bfloat16 resolution (~0.008), so it never
        # affects genuine confidence rankings.
        tiebreak = torch.arange(L, device=device, dtype=torch.float32) * 1e-7
        _, top_indices = torch.topk(confidence.float() - tiebreak, k=n_unmask, largest=True)
        new_response = torch.full((L,), mask_token_id, dtype=torch.long, device=device)
        new_response[top_indices] = pred_ids[top_indices]

        input_ids[0, prompt_len:] = new_response

    # Final response (drop everything after the first EOS if present)
    final = input_ids[0, prompt_len:]
    eos_positions = (final == eos_token_id).nonzero(as_tuple=True)[0]
    if len(eos_positions) > 0:
        final = final[: eos_positions[0]]

    return final

def perform_sampling_step(model, input_ids, mask_positions, config, step):
    """
    Example representation of the sampling step inside your generation loop.
    """
    # 1. Forward pass to get base logits
    outputs = model(input_ids)
    logits = outputs.logits
    
    if config.use_pcg:
        if config.debug_logs and step % 10 == 0:
            logger.info(f"--- Step {step} | PCG ACTIVE ---")
            
        # A. Classifier-Free Guidance (CFG) via Position Uncertainty
        # Calculate mean across the sequence length (dim=1)
        mean_logits = logits.mean(dim=1, keepdim=True)
        
        # Log to verify the shape is broadcastable (Batch, 1, Vocab)
        if config.debug_logs and step % 10 == 0:
            logger.info(f"Base logits shape: {logits.shape}, Mean logits shape: {mean_logits.shape}")
            logger.info(f"Sample logit variance before CFG: {logits.var().item():.4f}")
            
        # Push logits away from the mean position-agnostic distribution
        logits = logits + config.pcg_cfg_weight * (logits - mean_logits)
        
        if config.debug_logs and step % 10 == 0:
            logger.info(f"Sample logit variance after CFG (Weight {config.pcg_cfg_weight}): {logits.var().item():.4f}")

    # 2. Calculate probabilities and confidence
    probs = torch.softmax(logits, dim=-1)
    confidence, predictions = torch.max(probs, dim=-1)

    if config.use_pcg:
        # B. Soft Left-to-Right (SLR) Bias
        seq_len = confidence.shape[1]
        position_penalty = torch.arange(seq_len, device=confidence.device).float() / seq_len
        
        # Scale penalty by SLR weight
        scaled_penalty = config.pcg_slr_weight * position_penalty.unsqueeze(0)
        
        if config.debug_logs and step % 10 == 0:
            logger.info(f"SLR Penalty | Min: {scaled_penalty.min().item():.4f}, Max: {scaled_penalty.max().item():.4f}")
            logger.info(f"Avg Confidence BEFORE SLR: {confidence.mean().item():.4f}")
            
        # Apply penalty to confidence scores (penalizes tokens further to the right)
        confidence = confidence - scaled_penalty
        
        if config.debug_logs and step % 10 == 0:
            logger.info(f"Avg Confidence AFTER SLR: {confidence.mean().item():.4f}")

    # ... [Rest of your original code: sort `confidence`, pick bottom N%, and remask] ...
    # return updated_input_ids
