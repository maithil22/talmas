"""
TALMAS: Timestep-Adaptive, Layer-Dependent Masked Attention Suppression.

Implements the gate functions and hook manager described in the TALMAS proposal.
The hook manager patches each LLaDALlamaBlock's forward() to inject a logit bias
into attention_bias before the attention call — no model weights are modified.

Suppression regime:
  Query \ Key  |  Real (m_j=0)  |  [MASK] (m_j=1)
  -------------|----------------|------------------
  Real (m_i=0) |  no bias       |  -λ         (full suppression)
  [MASK](m_i=1)|  no bias       |  -λ·μ       (partial; preserves coordination)

λ(ℓ, t) = λ_max · f(1-r_t) · g(ℓ/L)
  f(x) = x²               (timestep gate: grows as more tokens are revealed)
  g(u) = sigmoid(8(u-0.5)) (layer gate: peaks at deep layers)
"""

import functools
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.config import TALMASConfig


# ---------------------------------------------------------------------------
# Gate functions
# ---------------------------------------------------------------------------

def f_timestep(x: float, exponent: float = 2.0) -> float:
    """Timestep gate.  x = 1 - r_t (fraction of tokens revealed)."""
    return x ** exponent


def g_layer(layer_idx: int, num_layers: int, slope: float = 8.0) -> float:
    """Sigmoid layer gate.  Peaks at deep layers."""
    u = layer_idx / num_layers
    return torch.sigmoid(torch.tensor(slope * (u - 0.5))).item()


def compute_lambda(
    lambda_max: float,
    r_t: float,
    layer_idx: int,
    num_layers: int,
    use_timestep_gate: bool = True,
    use_layer_gate: bool = True,
    sigmoid_slope: float = 8.0,
    timestep_exponent: float = 2.0,
) -> float:
    """
    λ(ℓ, t) = λ_max · f(1-r_t) · g(ℓ/L)

    r_t: current mask ratio (fraction of tokens still masked), 0 ≤ r_t ≤ 1.
    Gates can be disabled independently for ablation.
    """
    x = 1.0 - r_t  # fraction revealed
    f = f_timestep(x, exponent=timestep_exponent) if use_timestep_gate else 1.0
    g = g_layer(layer_idx, num_layers, slope=sigmoid_slope) if use_layer_gate else 1.0
    return lambda_max * f * g


# ---------------------------------------------------------------------------
# Hook manager
# ---------------------------------------------------------------------------

class TALMASHookManager:
    """
    Patches each LLaDALlamaBlock's forward() to inject the TALMAS asymmetric
    logit bias into attention_bias before the attention call.

    Usage:
        manager = TALMASHookManager(model, cfg)
        for step in denoising_loop:
            manager.set_state(r_t=..., mask_positions=...)
            output = model(input_ids=...)
        manager.remove()
    """

    def __init__(self, model, cfg: TALMASConfig, debug_step: Optional[int] = None):
        self.model = model
        self.cfg = cfg
        self.debug_step = debug_step

        # Runtime state — updated each diffusion step via set_state()
        self.r_t: float = 1.0
        self.step_idx: int = 0
        self.mask_positions: Optional[torch.Tensor] = None  # (batch, seq_len) bool
        self._debug_done: bool = False  # fire at most once per run

        self._patched = []                              # list of (module, original_forward)
        self.num_layers = self._count_layers()
        self._register_patches()

    def _count_layers(self) -> int:
        if hasattr(self.model.config, "num_hidden_layers"):
            return self.model.config.num_hidden_layers
        # Fallback: count transformer blocks
        count = sum(
            1 for name, _ in self.model.named_modules()
            if "transformer.blocks." in name and name.count(".") == 3
        )
        return max(count, 1)

    def _patch_attention(self, block_module, layer_idx: int) -> None:
        """
        Patch LLaDALlamaBlock.forward to inject TALMAS bias into attention_bias.
        LLaDA passes attention_bias directly into self.attention(q, k, v, attention_bias),
        so we intercept at the block level and add our logit bias there.
        """
        manager = self
        original_forward = block_module.forward

        @functools.wraps(original_forward)
        def patched_forward(x, attention_bias=None, **kwargs):
            if manager.mask_positions is not None and manager.cfg.lambda_max > 0.0:
                lam = compute_lambda(
                    manager.cfg.lambda_max,
                    manager.r_t,
                    layer_idx,
                    manager.num_layers,
                    use_timestep_gate=manager.cfg.use_timestep_gate,
                    use_layer_gate=manager.cfg.use_layer_gate,
                    sigmoid_slope=manager.cfg.sigmoid_slope,
                    timestep_exponent=manager.cfg.timestep_exponent,
                )

                if lam > 0.0:
                    m = manager.mask_positions.float().to(x.device)  # (B, S)

                    m_key   = m.unsqueeze(1).unsqueeze(2)   # (B, 1, 1, S)
                    m_query = m.unsqueeze(1).unsqueeze(3)   # (B, 1, S, 1)
                    query_gate = (1.0 - m_query) + manager.cfg.mu * m_query
                    talmas_bias = -(lam * m_key * query_gate)       # (B, 1, S, S)

                    # Match the dtype that _cast_attn_bias expects
                    talmas_bias = talmas_bias.to(dtype=x.dtype)

                    if attention_bias is not None:
                        attention_bias = attention_bias + talmas_bias
                    else:
                        attention_bias = talmas_bias

                    # Debug: capture attention values before and after suppression
                    # Fires once at the requested step, last layer only.
                    should_debug = (
                        not manager._debug_done
                        and manager.debug_step is not None
                        and manager.step_idx == manager.debug_step
                        and layer_idx == manager.num_layers - 1
                    )
                    if should_debug:
                        _capture_and_print_debug(
                            original_forward, x, attention_bias,
                            manager.mask_positions, manager.step_idx, manager.r_t, lam,
                            **kwargs,
                        )
                        manager._debug_done = True

            return original_forward(x, attention_bias=attention_bias, **kwargs)

        block_module.forward = patched_forward
        self._patched.append((block_module, original_forward))

    def _register_patches(self) -> None:
        """Patch each LLaDALlamaBlock — named model.transformer.blocks.N"""
        layer_idx = 0
        for name, module in self.model.named_modules():
            if "transformer.blocks." in name and name.count(".") == 3:
                self._patch_attention(module, layer_idx)
                layer_idx += 1

    def set_state(self, r_t: float, mask_positions: torch.Tensor, step_idx: int = 0) -> None:
        """Call once per diffusion step before the forward pass."""
        self.r_t = r_t
        self.step_idx = step_idx
        self.mask_positions = mask_positions

    def remove(self) -> None:
        """Restore all original block forward methods."""
        for block_module, original_forward in self._patched:
            block_module.forward = original_forward
        self._patched.clear()
        self.mask_positions = None


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def _capture_and_print_debug(
    original_forward,
    x: torch.Tensor,
    attention_bias_with_talmas,   # already has talmas_bias added
    mask_positions: torch.Tensor, # (B, S) bool
    step_idx: int,
    r_t: float,
    lam: float,
    **kwargs,
) -> None:
    """
    Run one extra forward pass through the last block with F.sdpa patched to
    capture raw logits and attention weights both with and without the TALMAS
    bias.  The extra pass is inference-mode only and its output is discarded.
    """
    debug: dict = {}
    old_sdpa = F.scaled_dot_product_attention

    def _capturing_sdpa(query, key, value, attn_mask=None, **kw):
        out = old_sdpa(query, key, value, attn_mask=attn_mask, **kw)
        scale = query.shape[-1] ** -0.5
        with torch.no_grad():
            raw_logits = (
                torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
            )  # (B, H, S, S)  — pure QK, no bias at all
            biased_logits = (
                raw_logits + attn_mask.float() if attn_mask is not None else raw_logits
            )  # (B, H, S, S)  — with causal + TALMAS bias
            debug["raw_logits"]    = raw_logits.cpu()
            debug["biased_logits"] = biased_logits.cpu()
            debug["weights_raw"]   = torch.softmax(raw_logits,    dim=-1).cpu()
            debug["weights_biased"]= torch.softmax(biased_logits, dim=-1).cpu()
        return out

    F.scaled_dot_product_attention = _capturing_sdpa
    try:
        with torch.no_grad():
            original_forward(x, attention_bias=attention_bias_with_talmas, **kwargs)
    finally:
        F.scaled_dot_product_attention = old_sdpa

    if debug:
        _print_attention_debug(
            debug["raw_logits"], debug["biased_logits"],
            debug["weights_raw"], debug["weights_biased"],
            mask_positions[0].cpu().numpy(), step_idx, r_t, lam,
        )


def _print_attention_debug(
    raw_logits: torch.Tensor,       # (B, H, S, S)  pre-softmax, no bias
    biased_logits: torch.Tensor,    # (B, H, S, S)  pre-softmax, with causal + TALMAS bias
    weights_raw: torch.Tensor,      # (B, H, S, S)  post-softmax, no bias
    weights_biased: torch.Tensor,   # (B, H, S, S)  post-softmax, with bias
    mask: np.ndarray,               # (S,) bool  True = [MASK] token
    step_idx: int,
    r_t: float,
    lam: float,
) -> None:
    # Mean over batch and heads → (S, S)
    rl = raw_logits.mean(dim=(0, 1)).float().numpy()
    bl = biased_logits.mean(dim=(0, 1)).float().numpy()
    wr = weights_raw.mean(dim=(0, 1)).float().numpy()
    wb = weights_biased.mean(dim=(0, 1)).float().numpy()

    S = rl.shape[0]
    m = mask.astype(bool)
    r = ~m
    m_q, m_k = m[:, np.newaxis], m[np.newaxis, :]
    r_q, r_k = r[:, np.newaxis], r[np.newaxis, :]

    quadrants = [
        ("real  → real  ", r_q & r_k),
        ("real  → [MASK]", r_q & m_k),
        ("[MASK] → real  ", m_q & r_k),
        ("[MASK] → [MASK]", m_q & m_k),
    ]

    def stats(arr: np.ndarray, sel: np.ndarray) -> str:
        vals = arr[sel]
        if vals.size == 0:
            return "                  n/a                  "
        return f"mean={vals.mean():+10.6f}  max={vals.max():+10.6f}"

    W = 104
    print(f"\n{'='*W}")
    print(f"TALMAS attention debug — step {step_idx}  r_t={r_t:.4f}  λ={lam:.4f}  "
          f"last layer  ({int(r.sum())} real / {int(m.sum())} [MASK], S={S})")
    print(f"{'='*W}")

    hdr = f"  {'Quadrant':<18}  {'pre-softmax (no bias)':^39}  {'pre-softmax (w/ suppression)':^39}"
    print(hdr)
    print(f"  {'-'*18}  {'-'*39}  {'-'*39}")
    for name, sel in quadrants:
        print(f"  {name:<18}  {stats(rl, sel)}  {stats(bl, sel)}")

    print()
    hdr2 = f"  {'Quadrant':<18}  {'post-softmax (no bias)':^39}  {'post-softmax (w/ suppression)':^39}"
    print(hdr2)
    print(f"  {'-'*18}  {'-'*39}  {'-'*39}")
    for name, sel in quadrants:
        print(f"  {name:<18}  {stats(wr, sel)}  {stats(wb, sel)}")
    print(f"{'='*W}\n")