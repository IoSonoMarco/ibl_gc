import numpy as np
import scipy

def poisson_loglik(y, mu, eps=1e-12):
    mu = np.clip(mu, eps, None)
    return np.sum(y * np.log(mu) - mu - scipy.special.gammaln(y + 1))