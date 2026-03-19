import numpy as np
import pandas as pd
from lib.models.utils import sigmoid
from lib.models.response_model import contrast_to_sensory_reliability


### Standard HGF 2-level ###

def hgf_binary_2level(
    u,
    omega2=-3.0,   # tonic log-volatility at level 2; process variance = exp(omega2)
    clip_p=1e-6,   # numerical stability for log
):
    """
    Minimal 2-level binary HGF forward pass (Mathys et al., 2014) with pyhgf-like outputs.

    Generative model (2-level reduction):
      Level 2 latent (logit of p):
        x2(k) = x2(k-1) + w2(k),   w2(k) ~ N(0, exp(omega2))

      Level 1 observation:
        u(k) ~ Bernoulli( sigmoid(x2(k)) )

    Filtering (Gaussian/Laplace over x2):
      Prediction:
        mu2_hat = mu2_prev
        pi2_hat = 1 / (1/pi2_prev + exp(omega2))

      Bernoulli prediction:
        p_hat = sigmoid(mu2_hat)

      Observation surprisal:
        s = -log p(u | p_hat)
          = -(u*log(p_hat) + (1-u)*log(1-p_hat))

      Update:
        pe = u - p_hat
        pi2 = pi2_hat + p_hat*(1-p_hat)     (curvature term)
        mu2 = mu2_hat + 1/pi2 * pe

    Notes on pyhgf-style bookkeeping:
      - `x_2_expected_*` are priors (hats), `x_2_*` are posteriors.
      - For binary nodes, `observation_input_0_expected_precision` corresponds to Bernoulli variance p(1-p).
    """
    u = np.asarray(u, dtype=float)
    n = u.size

    # Init
    mu2_0 = 0. # initial belief about log-odds of stimulus (pyhgf: initial_mean={"2": 0.0})
    pi2_0 = 1. # initial certainty about that belief (pyhgf: initial_precision={"2": 0.0})

    # Allocate
    mu2_hat = np.zeros(n)
    pi2_hat = np.zeros(n)
    mu2 = np.zeros(n)
    pi2 = np.zeros(n)

    p_hat = np.zeros(n)       # x_1_expected_mean
    var1_hat = np.zeros(n)    # observation_input_0_expected_precision (Bernoulli variance)
    surprise_u = np.zeros(n)  # observation_input_0_surprise

    mu2_prev = float(mu2_0)
    pi2_prev = float(pi2_0)

    q = np.exp(omega2)  # process variance increment per trial (since dt=1)

    for k in range(n):
        # ---- prediction (level 2) ----
        mu2_hat[k] = mu2_prev
        pi2_hat[k] = 1.0 / ((1.0 / max(pi2_prev, 1e-16)) + q)

        # ---- prediction (level 1) ----
        p = sigmoid(mu2_hat[k])
        p = np.clip(p, clip_p, 1.0 - clip_p)
        p_hat[k] = p

        var = p * (1.0 - p)
        var1_hat[k] = var

        # ---- observation surprise ----
        surprise_u[k] = -(u[k] * np.log(p) + (1.0 - u[k]) * np.log(1.0 - p))

        # ---- update (level 2) ----
        pe = u[k] - p
        pi2[k] = pi2_hat[k] + var
        mu2[k] = mu2_hat[k] + 1/pi2[k] * pe

        mu2_prev, pi2_prev = mu2[k], pi2[k]

    return pd.DataFrame(
        {
            "time_steps": np.arange(n, dtype=int),
            "observation_input_0": u,

            "x_1_expected_mean": p_hat,
            "observation_input_0_expected_precision": var1_hat,
            "observation_input_0_surprise": surprise_u,

            "x_2_expected_mean": mu2_hat,
            "x_2_expected_precision": pi2_hat,
            "x_2_mean": mu2,
            "x_2_precision": pi2,
        }
    )
    
    
### HGF with contrast-weight for precision at level-2 ###

def hgf_binary_2level_contrast_gate(
    stimulus_side, stimulus_contrast,
    omega2: float = -3.0,   # tonic log-volatility at level 2; process variance = exp(omega2)
    contrast_slope: float = 2, # Naka-Rushton exponential for the psychometric curve
    contrast_midpoint: float = 0.125, # Naka-Rushton midpoint for the psychometric curve
    clip_p: float = 1e-6,   # numerical stability for log
):
    """
    Minimal 2-level binary HGF forward pass (Mathys et al., 2014) with pyhgf-like outputs.

    Generative model (2-level reduction):
      Level 2 latent (logit of p):
        x2(k) = x2(k-1) + w2(k),   w2(k) ~ N(0, exp(omega2))

      Level 1 observation:
        u(k) ~ Bernoulli( sigmoid(x2(k)) )

    Filtering (Gaussian/Laplace over x2):
      Prediction:
        mu2_hat = mu2_prev
        pi2_hat = 1 / (1/pi2_prev + exp(omega2))

      Bernoulli prediction:
        p_hat = sigmoid(mu2_hat)

      Observation surprisal:
        s = -log p(u | p_hat)
          = -(u*log(p_hat) + (1-u)*log(1-p_hat))

      Update:
        pe = u - p_hat
        pi2 = pi2_hat + p_hat*(1-p_hat)     (curvature term)
        mu2 = mu2_hat + 1/pi2 * pe

    Notes on pyhgf-style bookkeeping:
      - `x_2_expected_*` are priors (hats), `x_2_*` are posteriors.
      - For binary nodes, `observation_input_0_expected_precision` corresponds to Bernoulli variance p(1-p).
    """
    u = stimulus_side
    w = contrast_to_sensory_reliability(stimulus_contrast, contrast_slope, contrast_midpoint)
    n = u.size

    # Init
    mu2_0 = 0. # initial belief about log-odds of stimulus (pyhgf: initial_mean={"2": 0.0})
    pi2_0 = 1. # initial certainty about that belief (pyhgf: initial_precision={"2": 0.0})

    # Allocate
    mu2_hat = np.zeros(n)
    pi2_hat = np.zeros(n)
    mu2 = np.zeros(n)
    pi2 = np.zeros(n)

    p_hat = np.zeros(n)       # x_1_expected_mean
    var1_hat = np.zeros(n)    # observation_input_0_expected_precision (Bernoulli variance)
    surprise_u = np.zeros(n)  # observation_input_0_surprise
    prediction_error = np.zeros(n)

    mu2_prev = float(mu2_0)
    pi2_prev = float(pi2_0)

    q = np.exp(omega2)  # process variance increment per trial (since dt=1)

    for k in range(n):
        # ---- prediction (level 2) ----
        mu2_hat[k] = mu2_prev
        pi2_hat[k] = 1.0 / ((1.0 / max(pi2_prev, 1e-16)) + q)

        # ---- prediction (level 1) ----
        p = sigmoid(mu2_hat[k])
        p = np.clip(p, clip_p, 1.0 - clip_p)
        p_hat[k] = p

        var = p * (1.0 - p)
        var1_hat[k] = var

        # ---- observation surprise ----
        surprise_u[k] = -(u[k] * np.log(p) + (1.0 - u[k]) * np.log(1.0 - p))

        # ---- update (level 2) ----
        pe = u[k] - p
        prediction_error[k] = pe
        pi2[k] = pi2_hat[k] + var * w[k]
        mu2[k] = mu2_hat[k] + 1/pi2[k] * pe * w[k]

        mu2_prev, pi2_prev = mu2[k], pi2[k]

    return pd.DataFrame(
        {
            "time_steps": np.arange(n, dtype=int),
            "observation_input_0": u,

            "x_1_expected_mean": p_hat,
            "observation_input_0_expected_precision": var1_hat,
            "observation_input_0_surprise": surprise_u,
            "observation_input_0_pe": prediction_error,
            "observation_input_0_weighted_pe": prediction_error * w,

            "x_2_expected_mean": mu2_hat,
            "x_2_expected_precision": pi2_hat,
            "x_2_mean": mu2,
            "x_2_precision": pi2,

            "w_reliability": w,
        }
    )