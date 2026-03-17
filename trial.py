from cmdstanpy import CmdStanModel
import numpy as np

if __name__ == "__main__":
    
    true_mu = 1.5
    true_sigma = 3
    N = 200
    y = np.random.normal(true_mu, true_sigma, size=N)
    
    model = CmdStanModel(stan_file="./trial.stan")
    data = {
        "N": N,
        "y": y.tolist()
    }

    fit = model.sample(
        data=data,
        chains=2,
        iter_warmup=200,
        iter_sampling=1000
    )
    
    fit.draws_pd().to_csv("./draws.csv", index=False)