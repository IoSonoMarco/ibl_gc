import os
import numpy as np
from enum import Enum
import pandas as pd
from lib.filepaths import ROOT_STAN_MODELS

def sigmoid(x, safe=True):
    x = np.clip(x, -40, 40) if safe else x
    return 1.0 / (1.0 + np.exp(-x))

def gaussian_kl_1d(mu_q, pi_q, mu_p, pi_p, eps=1e-16):
    """
    KL( q || p ) for 1D Gaussians:
      q = N(mu_q, 1/pi_q), p = N(mu_p, 1/pi_p)
    """
    pi_q = np.maximum(pi_q, eps)
    pi_p = np.maximum(pi_p, eps)

    var_q = 1.0 / pi_q
    var_p = 1.0 / pi_p

    return 0.5 * (
        np.log(var_p / var_q)
        + (var_q + (mu_q - mu_p) ** 2) / var_p
        - 1.0
    )

###

class PerceptualModule(Enum):
    BASIC="basic" # standard HGF belief updating
    SCPW="scpw" # stimulus contrast precision weight
    
class ResponseModule(Enum):
    BASIC="basic" # use HGF bottom-level stimulus probs
    BRT="brt" # belief-reliability trade-off
    BRRT="brrt" # belief-reliability-reward trade-off

def get_stan_model_paths():
    stan_model_paths = pd.DataFrame([
        dict(perc=PerceptualModule.SCPW, resp=ResponseModule.BASIC, filename="perc__stimulus_contrast_precision_weight-res__basic.stan"),
        dict(perc=PerceptualModule.SCPW, resp=ResponseModule.BRT, filename="perc__stimulus_contrast_precision_weight-res__belief_reliability_tradeoff.stan"),
        dict(perc=PerceptualModule.SCPW, resp=ResponseModule.BRRT, filename="perc__stimulus_contrast_precision_weight-res__belief_reliability_reward_tradeoff.stan"),
    ])
    stan_model_paths["single_session_root"] = os.path.join(ROOT_STAN_MODELS, "single_session") 
    stan_model_paths["single_subject_root"] = os.path.join(ROOT_STAN_MODELS, "single_subject")
    stan_model_paths["full_root"] = os.path.join(ROOT_STAN_MODELS, "full")
    return stan_model_paths