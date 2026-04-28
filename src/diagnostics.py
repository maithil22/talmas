"""
Diagnostic data collection and visualization for TALMAS denoising analysis.

Captures per-step data during low_confidence_remasking_sample and writes
four files to output_dir/:
  attention.gif    — animated attention heatmap (last layer, response×response)
  suppression.gif  — animated TALMAS bias matrix (last layer, response×response)
  confidence.png   — per-position confidence at sampled denoising steps
  scalar.png       — mean attention flowing to [MASK] tokens over all steps

Install DiagnosticsCollector AFTER TALMASHookManager so it wraps the
already-patched block forward.  Remove it BEFORE TALMASHookManager.remove().
"""

import functools
import io
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.config import TALMASConfig
from src.talmas import compute_lambda


class DiagnosticsCollector:

    def __init__(
        self,
        model,
        talmas_cfg: Optional[TALMASConfig],
        num_layers: int,
        capture_attn_every: int = 10,
        capture_conf_every: int = 50,
    ):
        self.talmas_cfg = talmas_cfg
        self.num_layers = num_layers
        self.capture_attn_every = capture_attn_every
        self.capture_conf_every = capture_conf_every

        # Per-step storage
        self._attn: Dict[int, np.ndarray] = {}   # step → (S, S) mean-over-heads
        self._supp: Dict[int, np.ndarray] = {}   # step → (S, S) bias magnitude
        self._conf: Dict[int, np.ndarray] = {}   # step → (L,)  post-clamp confidence
        self._mask: Dict[int, np.ndarray] = {}   # step → (S,)  bool
        self._t_vals: Dict[int, float] = {}       # step → r_t

        self._should_capture = False
        self._latest_attn: Optional[np.ndarray] = None
        self._last_block = None
        self._orig_fwd = None

        self._install_capture(model)

    # ------------------------------------------------------------------
    # Hook installation
    # ------------------------------------------------------------------

    def _install_capture(self, model) -> None:
        last = None
        for name, module in model.named_modules():
            if "transformer.blocks." in name and name.count(".") == 3:
                last = module
        if last is None:
            print("DiagnosticsCollector: last block not found — attention capture disabled")
            return

        self._last_block = last
        self._orig_fwd = last.forward
        collector = self

        @functools.wraps(self._orig_fwd)
        def capturing_fwd(x, attention_bias=None, **kwargs):
            if not collector._should_capture:
                return collector._orig_fwd(x, attention_bias=attention_bias, **kwargs)

            captured: List[np.ndarray] = []
            old_sdpa = F.scaled_dot_product_attention

            def capturing_sdpa(query, key, value, attn_mask=None, **kw):
                # Call real F.sdpa so model output is unchanged
                out = old_sdpa(query, key, value, attn_mask=attn_mask, **kw)
                # Recompute attention weights separately for visualization only
                scale = query.shape[-1] ** -0.5
                with torch.no_grad():
                    logits = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
                    if attn_mask is not None:
                        logits = logits + attn_mask.float()
                    weights = torch.softmax(logits, dim=-1)      # (B, H, S, S)
                    captured.append(
                        weights.mean(dim=1)[0].cpu().numpy().astype(np.float32)  # (S, S)
                    )
                return out

            F.scaled_dot_product_attention = capturing_sdpa
            try:
                result = collector._orig_fwd(x, attention_bias=attention_bias, **kwargs)
            finally:
                F.scaled_dot_product_attention = old_sdpa

            if captured:
                collector._latest_attn = captured[0]
            return result

        last.forward = capturing_fwd

    # ------------------------------------------------------------------
    # Sampling loop interface — called from low_confidence_remasking_sample
    # ------------------------------------------------------------------

    def begin_step(self, step_idx: int, t_val: float, mask_positions: torch.Tensor) -> None:
        """Call BEFORE the model forward pass. Arms attention capture and computes suppression."""
        self._t_vals[step_idx] = t_val
        self._latest_attn = None

        m = mask_positions[0].cpu().numpy()   # (S,) bool
        self._mask[step_idx] = m

        capture = (step_idx % self.capture_attn_every == 0)
        self._should_capture = capture

        # Compute suppression bias analytically for the last layer at this step
        if capture and self.talmas_cfg is not None and self.talmas_cfg.lambda_max > 0.0:
            lam = compute_lambda(
                self.talmas_cfg.lambda_max, t_val,
                self.num_layers - 1, self.num_layers,
                use_timestep_gate=self.talmas_cfg.use_timestep_gate,
                use_layer_gate=self.talmas_cfg.use_layer_gate,
                sigmoid_slope=self.talmas_cfg.sigmoid_slope,
                timestep_exponent=self.talmas_cfg.timestep_exponent,
            )
            mf = m.astype(np.float32)
            query_gate = (1.0 - mf[:, None]) + self.talmas_cfg.mu * mf[:, None]  # (S, 1)
            self._supp[step_idx] = (lam * mf[None, :] * query_gate).astype(np.float32)

    def end_step(self, step_idx: int, confidence: torch.Tensor) -> None:
        """Call AFTER confidence is finalized (post zero-EOS and already-unmasked clamping)."""
        self._should_capture = False
        if self._latest_attn is not None:
            self._attn[step_idx] = self._latest_attn
        if step_idx % self.capture_conf_every == 0:
            self._conf[step_idx] = confidence.detach().cpu().float().numpy()

    def remove(self) -> None:
        if self._last_block is not None and self._orig_fwd is not None:
            self._last_block.forward = self._orig_fwd

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_all(self, prompt_len: int, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nGenerating diagnostics in {output_dir}/")

        if self._attn:
            _make_gif(
                self._attn, self._mask, self._t_vals, prompt_len,
                title_prefix="Attention (last layer, mean over heads)",
                cmap="viridis",
                out_path=os.path.join(output_dir, "attention.gif"),
            )
            _plot_scalar(
                self._attn, self._mask, self._t_vals, prompt_len,
                out_path=os.path.join(output_dir, "scalar.png"),
            )

        if self._supp:
            _make_gif(
                self._supp, self._mask, self._t_vals, prompt_len,
                title_prefix="TALMAS suppression bias (last layer)",
                cmap="Reds",
                out_path=os.path.join(output_dir, "suppression.gif"),
            )

        if self._conf:
            _plot_confidence(
                self._conf, self._mask, prompt_len,
                out_path=os.path.join(output_dir, "confidence.png"),
            )


# ---------------------------------------------------------------------------
# Private plot helpers
# ---------------------------------------------------------------------------

def _resp(arr: np.ndarray, p: int) -> np.ndarray:
    """Crop (S, S) matrix to the response×response region."""
    return arr[p:, p:]


def _fig_to_pil(fig):
    """Render a matplotlib figure to a PIL Image via an in-memory PNG buffer."""
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    return img


def _make_gif(
    data: Dict[int, np.ndarray],
    mask_data: Dict[int, np.ndarray],
    t_vals: Dict[int, float],
    prompt_len: int,
    title_prefix: str,
    cmap: str,
    out_path: str,
    fps: int = 8,
) -> None:
    try:
        from PIL import Image
    except ImportError:
        print(f"  pillow not installed — skipping {os.path.basename(out_path)}")
        return

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    steps = sorted(data.keys())
    resp_frames = [_resp(data[s], prompt_len) for s in steps]
    vmax = max(f.max() for f in resp_frames) or 1.0
    L = resp_frames[0].shape[0]
    tick_step = max(32, L // 8)
    ticks = list(range(0, L, tick_step))

    pil_frames = []
    for i, step in enumerate(steps):
        arr = resp_frames[i]
        mask = mask_data[step][prompt_len:]   # (L,) bool — response positions
        n_masked = int(mask.sum())
        r_t = t_vals[step]

        fig = plt.figure(figsize=(6.5, 7.5))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 22], hspace=0.03)

        # Top strip: mask indicator for key/query positions (same tokens on both axes)
        ax_strip = fig.add_subplot(gs[0])
        ax_strip.imshow(
            mask[None, :].astype(np.float32),
            aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1, interpolation="none",
        )
        ax_strip.set_xticks([])
        ax_strip.set_yticks([0])
        ax_strip.set_yticklabels(["[MASK]"], fontsize=6)
        ax_strip.set_title(
            f"{title_prefix}\nstep {step}   r_t={r_t:.3f}   {n_masked}/{L} masked",
            fontsize=8,
        )

        # Main heatmap: response × response
        ax = fig.add_subplot(gs[1])
        im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks, fontsize=6)
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticks, fontsize=6)
        ax.set_xlabel("Key (response position)", fontsize=7)
        ax.set_ylabel("Query (response position)", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

        pil_frames.append(_fig_to_pil(fig))
        plt.close(fig)

    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=1000 // fps,
        loop=0,
    )
    print(f"  {os.path.basename(out_path)}: {len(pil_frames)} frames @ {fps} fps")


def _plot_confidence(
    conf_data: Dict[int, np.ndarray],
    mask_data: Dict[int, np.ndarray],
    prompt_len: int,
    out_path: str,
) -> None:
    import matplotlib.pyplot as plt

    steps = sorted(conf_data.keys())
    L = len(conf_data[steps[0]])
    cmap = plt.get_cmap("plasma")
    xs = np.arange(L)

    fig, ax = plt.subplots(figsize=(13, 4))
    for i, step in enumerate(steps):
        color = cmap(i / max(len(steps) - 1, 1))
        ax.plot(xs, conf_data[step], color=color, linewidth=0.9, alpha=0.85, label=f"step {step}")

    ax.axhline(1.0, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_xlim(0, L - 1)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlabel("Response position")
    ax.set_ylabel("Confidence")
    ax.set_title("Token confidence per response position  (light → dark = early → late steps)")
    ax.legend(fontsize=6, ncol=max(1, len(steps) // 2), loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  confidence.png saved")


def _plot_scalar(
    attn_data: Dict[int, np.ndarray],
    mask_data: Dict[int, np.ndarray],
    t_vals: Dict[int, float],
    prompt_len: int,
    out_path: str,
) -> None:
    import matplotlib.pyplot as plt

    steps = sorted(attn_data.keys())
    scalars: List[float] = []
    r_ts: List[float] = []

    for step in steps:
        attn = attn_data[step]           # (S, S)
        mask = mask_data[step]            # (S,) bool
        attn_resp = attn[prompt_len:]    # (L, S) — response queries only
        mask_resp = mask[prompt_len:]    # (L,) bool
        real_resp = ~mask_resp
        if real_resp.any() and mask.any():
            val = float(attn_resp[real_resp, :][:, mask].mean())
        else:
            val = 0.0
        scalars.append(val)
        r_ts.append(t_vals[step])

    t_map = dict(zip(steps, r_ts))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, scalars, "o-", color="#E87B4C", linewidth=1.5, markersize=3)
    ax.set_xlabel("Denoising step")
    ax.set_ylabel("Mean attention weight")
    ax.set_title(
        "Mean attention: real response tokens → [MASK] tokens  (last layer)\n"
        "With suppression this should be lower than baseline at the same step"
    )
    ax.set_ylim(bottom=0)

    # Secondary x-axis showing r_t
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    sampled = steps[:: max(1, len(steps) // 6)]
    ax2.set_xticks(sampled)
    ax2.set_xticklabels([f"{t_map[s]:.2f}" for s in sampled], fontsize=7)
    ax2.set_xlabel("r_t (mask ratio)", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  scalar.png saved")
