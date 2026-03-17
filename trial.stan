data {
  int<lower=0> N;
  array[N] real y;
}

parameters {
  real mu;
  real<lower=0> sigma;
}

model {
  // Priors
  target += normal_lpdf(mu | 0, 10);
  target += exponential_lpdf(sigma | 1);

  // Likelihood
  target += normal_lpdf(y | mu, sigma);
}

generated quantities {
  array[N] real y_rep;

  for (n in 1:N) {
    y_rep[n] = normal_rng(mu, sigma);
  }
}