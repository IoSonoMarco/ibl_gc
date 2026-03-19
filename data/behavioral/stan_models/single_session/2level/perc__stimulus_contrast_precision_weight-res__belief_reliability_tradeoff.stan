functions {

  real contrast_to_sensory_reliability(
    real contrast, 
    real contrast_slope,
    real contrast_midpoint
  ) {
      real c;
      real c50;

      c = pow(contrast, contrast_slope);
      c50 = pow(contrast_midpoint, contrast_slope);

      return c / (c + c50 + 1e-16);
    }

  vector perceptual_model(
    array[] int stimulus_side, 
    vector stimulus_contrast,
    real omega2,
    real contrast_slope,
    real contrast_midpoint
  ) {
    int N = size(stimulus_side);
    vector[N] m; // x_2_expected_mean

    // Initial states, matching the Python code
    real mu2_prev = 0.0;
    real pi2_prev = 1.0;

    real mu2_hat;
    real pi2_hat;
    real p_hat;
    real var1_hat;
    real pi2;
    real mu2;
    real pe;
    real w;

    // Process variance increment per trial
    real q = exp(omega2);

    for (n in 1:N) {
      // Contrast-derived reliability
      w = contrast_to_sensory_reliability(
        stimulus_contrast[n], 
        contrast_slope,
        contrast_midpoint
      );

      // Prediction (level 2)
      mu2_hat = mu2_prev;
      pi2_hat = inv(inv(pi2_prev) + q);
      m[n] = mu2_hat;

      // Prediction (level 1)
      p_hat = inv_logit(mu2_hat);
      var1_hat = p_hat * (1.0 - p_hat);

      // Update (level 2)
      pe = stimulus_side[n] - p_hat;
      pi2 = pi2_hat + var1_hat * w;
      mu2 = mu2_hat + (pe * w) / pi2;

      mu2_prev = mu2;
      pi2_prev = pi2;
    }

    return m;
  }

  vector response_model(
    vector mu2_hat,
    array[] int stimulus_side,
    vector stimulus_contrast,
    real beta_prior,
    real beta_sens,
    real contrast_slope,
    real contrast_midpoint,
    real lapse
  ) {
    int N = num_elements(mu2_hat);
    vector[N] p_choice;

    real sensory_reliability;
    real sensory_evidence;
    real theta;

    for (n in 1:N) {
      sensory_reliability = contrast_to_sensory_reliability(
        stimulus_contrast[n], 
        contrast_slope,
        contrast_midpoint
      );
      sensory_evidence = sensory_reliability * (2.0 * stimulus_side[n] - 1.0);

      theta = beta_prior*mu2_hat[n] + beta_sens*sensory_evidence;
      p_choice[n] = (1.0 - lapse) * inv_logit(theta) + 0.5 * lapse;
    }

    return p_choice;
  }
}

data {
  int<lower=1> N_obs; // num. observations
  array[N_obs] int<lower=0, upper=1> stimulus_side;
  vector<lower=0, upper=1>[N_obs] stimulus_contrast;
  array[N_obs] int<lower=-1, upper=1> choice;
}

parameters {
  real omega2_latent;       
  real beta_prior_latent;
  real beta_sens_latent;
  real contrast_slope_latent;
  real<lower=0, upper=1> contrast_midpoint;
  real lapse_latent;
}

transformed parameters {
  real<upper=0> omega2;
  real<lower=0> beta_prior;
  real<lower=0> beta_sens;
  real<lower=0.05, upper=5> contrast_slope;
  real<lower=0.001, upper=0.2> lapse;

  vector[N_obs] x_2_expected_mean;
  vector[N_obs] p_choice;

  omega2 = -log1p_exp(omega2_latent);
  beta_prior = exp(beta_prior_latent);
  beta_sens = exp(beta_sens_latent);
  contrast_slope = 0.5 + (5 - 0.5) * inv_logit(contrast_slope_latent);
  lapse = 0.001 + (0.2 - 0.001) * inv_logit(lapse_latent);

  // Perceptual -> Response Probs
  x_2_expected_mean = perceptual_model(
    stimulus_side, stimulus_contrast, 
    omega2, contrast_slope, contrast_midpoint
  );

  p_choice = response_model(
    x_2_expected_mean, 
    stimulus_side, 
    stimulus_contrast,
    beta_prior, beta_sens, 
    contrast_slope, contrast_midpoint,
    lapse
  );
}

model {

  // Prior
  target += std_normal_lpdf(omega2_latent);
  target += std_normal_lpdf(beta_prior_latent);
  target += std_normal_lpdf(beta_sens_latent);
  target += std_normal_lpdf(contrast_slope_latent);
  target += std_normal_lpdf(lapse_latent);  

  // Likelihood
  for (n in 1:N_obs) {
    if (choice[n] != -1)
        target += bernoulli_lpmf(choice[n] | p_choice[n]);
  }

}

generated quantities {
  vector[N_obs] log_lik;
  real brier_score = 0;
  real total_obs = 0;

  for (n in 1:N_obs) {

    if (choice[n] != -1) {

      log_lik[n] = bernoulli_lpmf(choice[n] | p_choice[n]);
      brier_score += square(choice[n] - p_choice[n]);
      total_obs += 1;
      
    } else {
      log_lik[n] = 0;
    }

  }

  brier_score /= total_obs;
}