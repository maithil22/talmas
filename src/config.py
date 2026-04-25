"""
Shared configuration dataclasses and presets for LLaDA evaluation and TALMAS ablation.
"""

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Sampling config  (verbatim from llada_gsm8k_eval.py)
# ---------------------------------------------------------------------------

@dataclass
class SamplingConfig:
    generation_length: int       # number of response tokens to generate
    steps: int                   # number of reverse diffusion steps (N)
    zero_eos_confidence: bool    # set EOS confidence=0 (needed for instruct model)
    few_shot: int = 4            # number of few-shot examples


BASE_CONFIG = SamplingConfig(
    generation_length=1024,
    steps=1024,
    zero_eos_confidence=False,
)

INSTRUCT_CONFIG = SamplingConfig(
    generation_length=512,
    steps=512,
    zero_eos_confidence=True,
)


# ---------------------------------------------------------------------------
# TALMAS config
# ---------------------------------------------------------------------------

@dataclass
class TALMASConfig:
    lambda_max: float = 4.0          # maximum logit suppression magnitude
    mu: float = 0.1                  # mask→mask partial suppression scale
    use_timestep_gate: bool = True   # apply quadratic f(1-r_t) gate
    use_layer_gate: bool = True      # apply sigmoid g(ℓ/L) layer gate
    sigmoid_slope: float = 8.0       # steepness of g(ℓ/L) sigmoid (Tier 3 sweep)
    timestep_exponent: float = 2.0   # exponent of f(x) = x^p (Tier 3 sweep)


# ---------------------------------------------------------------------------
# Ablation configurations  (5 configs from TALMAS proposal)
# ---------------------------------------------------------------------------

ABLATION_CONFIGS = [
    {
        "id": 1,
        "name": "Baseline (LLaDA)",
        "lambda_max": 0.0,
        "use_timestep_gate": False,
        "use_layer_gate": False,
        "mu": 0.0,
        "description": "λ_max=0 — recovers original LLaDA exactly",
    },
    {
        "id": 2,
        "name": "Static Bias",
        "lambda_max": 4.0,
        "use_timestep_gate": False,
        "use_layer_gate": False,
        "mu": 0.0,
        "description": "Fixed λ, no timestep or layer gating",
    },
    {
        "id": 3,
        "name": "Timestep-Only",
        "lambda_max": 4.0,
        "use_timestep_gate": True,
        "use_layer_gate": False,
        "mu": 0.0,
        "description": "Quadratic timestep ramp, uniform across layers",
    },
    {
        "id": 4,
        "name": "Layer-Only",
        "lambda_max": 4.0,
        "use_timestep_gate": False,
        "use_layer_gate": True,
        "mu": 0.0,
        "description": "Sigmoid layer gate, uniform across timesteps",
    },
    {
        "id": 5,
        "name": "Full TALMAS",
        "lambda_max": 4.0,
        "use_timestep_gate": True,
        "use_layer_gate": True,
        "mu": 0.1,
        "description": "Full joint gating with partial mask→mask suppression",
    },
]

MU_SWEEP = [0.0, 0.2, 0.5, 1.0]


# ---------------------------------------------------------------------------
# Sweep configs  (one entry = one hyperparameter combination)
#
# Tier 1 (IDs  1–25): λ_max × μ joint grid, Full TALMAS (both gates on)
# Tier 3 (IDs 26–28): sigmoid slope sweep — update lambda_max/mu to the best
#                      values found in Tier 1 before running these.
# ---------------------------------------------------------------------------

SWEEP_CONFIGS = [
    # --- Tier 1: λ_max × μ grid (Full TALMAS, both gates on, slope=8.0) ---
    {"id":  1, "tier": 1, "lambda_max": 1.0, "mu": 0.0, "sigmoid_slope": 8.0},
    {"id":  2, "tier": 1, "lambda_max": 1.0, "mu": 0.1, "sigmoid_slope": 8.0},
    {"id":  3, "tier": 1, "lambda_max": 1.0, "mu": 0.2, "sigmoid_slope": 8.0},
    {"id":  4, "tier": 1, "lambda_max": 1.0, "mu": 0.5, "sigmoid_slope": 8.0},
    {"id":  5, "tier": 1, "lambda_max": 1.0, "mu": 1.0, "sigmoid_slope": 8.0},
    {"id":  6, "tier": 1, "lambda_max": 2.0, "mu": 0.0, "sigmoid_slope": 8.0},
    {"id":  7, "tier": 1, "lambda_max": 2.0, "mu": 0.1, "sigmoid_slope": 8.0},
    {"id":  8, "tier": 1, "lambda_max": 2.0, "mu": 0.2, "sigmoid_slope": 8.0},
    {"id":  9, "tier": 1, "lambda_max": 2.0, "mu": 0.5, "sigmoid_slope": 8.0},
    {"id": 10, "tier": 1, "lambda_max": 2.0, "mu": 1.0, "sigmoid_slope": 8.0},
    {"id": 11, "tier": 1, "lambda_max": 4.0, "mu": 0.0, "sigmoid_slope": 8.0},
    {"id": 12, "tier": 1, "lambda_max": 4.0, "mu": 0.1, "sigmoid_slope": 8.0},
    {"id": 13, "tier": 1, "lambda_max": 4.0, "mu": 0.2, "sigmoid_slope": 8.0},
    {"id": 14, "tier": 1, "lambda_max": 4.0, "mu": 0.5, "sigmoid_slope": 8.0},
    {"id": 15, "tier": 1, "lambda_max": 4.0, "mu": 1.0, "sigmoid_slope": 8.0},
    {"id": 16, "tier": 1, "lambda_max": 6.0, "mu": 0.0, "sigmoid_slope": 8.0},
    {"id": 17, "tier": 1, "lambda_max": 6.0, "mu": 0.1, "sigmoid_slope": 8.0},
    {"id": 18, "tier": 1, "lambda_max": 6.0, "mu": 0.2, "sigmoid_slope": 8.0},
    {"id": 19, "tier": 1, "lambda_max": 6.0, "mu": 0.5, "sigmoid_slope": 8.0},
    {"id": 20, "tier": 1, "lambda_max": 6.0, "mu": 1.0, "sigmoid_slope": 8.0},
    {"id": 21, "tier": 1, "lambda_max": 8.0, "mu": 0.0, "sigmoid_slope": 8.0},
    {"id": 22, "tier": 1, "lambda_max": 8.0, "mu": 0.1, "sigmoid_slope": 8.0},
    {"id": 23, "tier": 1, "lambda_max": 8.0, "mu": 0.2, "sigmoid_slope": 8.0},
    {"id": 24, "tier": 1, "lambda_max": 8.0, "mu": 0.5, "sigmoid_slope": 8.0},
    {"id": 25, "tier": 1, "lambda_max": 8.0, "mu": 1.0, "sigmoid_slope": 8.0},
    # --- Tier 3: sigmoid slope sweep (update lambda_max/mu to best Tier 1 values) ---
    {"id": 26, "tier": 3, "lambda_max": 4.0, "mu": 0.1, "sigmoid_slope":  4.0},
    {"id": 27, "tier": 3, "lambda_max": 4.0, "mu": 0.1, "sigmoid_slope":  8.0},
    {"id": 28, "tier": 3, "lambda_max": 4.0, "mu": 0.1, "sigmoid_slope": 16.0},
]

# Lookup by id for O(1) access
SWEEP_CONFIG_BY_ID = {c["id"]: c for c in SWEEP_CONFIGS}
