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
    array[] int start_idx,
    array[] int end_idx,
    vector omega2,
    vector contrast_slope,
    vector contrast_midpoint
  ) {
    int N = size(stimulus_side);
    int K = size(start_idx);
    vector[N] m; // x_2_expected_mean

    // loop across sessions
    for (k in 1:K) {
        // Initial states (could also be estimated)
        real mu2_prev = 0.0;
        real pi2_prev = 1.0;
        real q = exp(omega2[k]); // omega2 for each session

        // Initial variables
        real mu2_hat;
        real pi2_hat;
        real p_hat;
        real var1_hat;
        real pi2;
        real mu2;
        real pe;
        real w;

        for (n in start_idx[k]:end_idx[k]) {
            // Contrast-derived reliability
            w = contrast_to_sensory_reliability(
              stimulus_contrast[n], 
              contrast_slope[k],
              contrast_midpoint[k]
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
    }
    return m;
  }

  vector response_model(
    vector mu2_hat,
    array[] int stimulus_side,
    vector stimulus_contrast,
    array[] int start_idx,
    array[] int end_idx,
    vector beta_prior,
    vector beta_sens,
    vector contrast_slope,
    vector contrast_midpoint,
    vector lapse
  ) {
    int N = num_elements(mu2_hat);
    int K = size(start_idx);
    vector[N] p_choice;
    
    // loop across sessions
    for (k in 1:K) {
        real beta_prior_k = beta_prior[k];
        real beta_sens_k = beta_sens[k];
        real contrast_slope_k = contrast_slope[k];
        real contrast_midpoint_k = contrast_midpoint[k];
        real lapse_k = lapse[k];

        real sensory_reliability;
        real sensory_evidence;
        real theta;

        for (n in start_idx[k]:end_idx[k]) {
            sensory_reliability = contrast_to_sensory_reliability(
                stimulus_contrast[n], 
                contrast_slope_k,
                contrast_midpoint_k
              );
            sensory_evidence = sensory_reliability * (2.0 * stimulus_side[n] - 1.0);

            theta = beta_prior_k*mu2_hat[n] + beta_sens_k*sensory_evidence;
            p_choice[n] = (1.0 - lapse_k) * inv_logit(theta) + 0.5 * lapse_k;
        }
    }
    return p_choice;
  }
}

data {
  int<lower=1> N_obs; // num. total observations (stacked for all sessions)
  int<lower=1> N_sess; // num. sessions

  array[N_obs] int<lower=0, upper=1> stimulus_side;
  vector<lower=0, upper=1>[N_obs] stimulus_contrast;
  array[N_obs] int<lower=-1, upper=1> choice;

  array[N_sess] int<lower=1> start_idx;
  array[N_sess] int<lower=1> end_idx;
}

parameters {
  vector[N_sess] omega2_eta;
  vector[N_sess] beta_prior_eta;
  vector[N_sess] beta_sens_eta;
  real contrast_slope_latent;
  real<lower=0, upper=1> contrast_midpoint;
  real lapse_latent;

  real mu_omega2;
  real<lower=0> sigma_omega2;

  real mu_beta_prior;
  real<lower=0> sigma_beta_prior;

  real mu_beta_sens;
  real<lower=0> sigma_beta_sens;
}

transformed parameters {
  vector<upper=0>[N_sess] omega2_sess;
  vector<lower=0>[N_sess] beta_prior_sess;
  vector<lower=0>[N_sess] beta_sens_sess;
  real<lower=0.5, upper=5> contrast_slope;
  real<lower=0.001, upper=0.2> lapse;

  omega2_sess = -log1p_exp(mu_omega2 + sigma_omega2 * omega2_eta);
  beta_prior_sess = exp(mu_beta_prior + sigma_beta_prior * beta_prior_eta);
  beta_sens_sess = exp(mu_beta_sens + sigma_beta_sens * beta_sens_eta);
  contrast_slope = 0.5 + (5 - 0.5) * inv_logit(contrast_slope_latent); // contrast_slope in [0.5, 5]
  lapse = 0.001 + (0.2 - 0.001) * inv_logit(lapse_latent); // lapse in [0.001, 0.2]


  vector[N_obs] x_2_expected_mean;
  vector[N_obs] p_choice;

  // Perceptual -> Response Probs
  x_2_expected_mean = perceptual_model(
    stimulus_side, stimulus_contrast, 
    start_idx, end_idx,
    omega2_sess, 
    rep_vector(contrast_slope, N_sess), 
    rep_vector(contrast_midpoint, N_sess)
  );

  p_choice = response_model(
    x_2_expected_mean, 
    stimulus_side, stimulus_contrast,
    start_idx, end_idx,
    beta_prior_sess, beta_sens_sess, 
    rep_vector(contrast_slope, N_sess), 
    rep_vector(contrast_midpoint, N_sess),
    rep_vector(lapse, N_sess)
  );
}

model {

  // Hyper-Priors
  target += std_normal_lpdf(mu_omega2);
  target += std_normal_lpdf(sigma_omega2);

  target += std_normal_lpdf(mu_beta_prior);
  target += std_normal_lpdf(sigma_beta_prior);

  target += std_normal_lpdf(mu_beta_sens);
  target += std_normal_lpdf(sigma_beta_sens);

  // Priors
  target += std_normal_lpdf(omega2_eta);
  target += std_normal_lpdf(beta_prior_eta);
  target += std_normal_lpdf(beta_sens_eta);

  target += std_normal_lpdf(contrast_slope_latent);
  target += std_normal_lpdf(lapse_latent);

  // Likelihood
  for (n in 1:N_obs) {
    if (choice[n] != -1)
        target += bernoulli_lpmf(choice[n] | p_choice[n]);
  }
  
}

generated quantities {
  vector[N_obs] log_lik = rep_vector(0, N_obs);
  vector[N_sess] log_lik_sess = rep_vector(0, N_sess);
  vector[N_sess] brier_score_sess = rep_vector(0, N_sess);
  vector[N_sess] n_obs_sess = rep_vector(0, N_sess);

  real brier_score = 0;
  real total_obs = 0;

  for (k in 1:N_sess) {
    for (n in start_idx[k]:end_idx[k]) {
      if (choice[n] != -1) {
        real sq_err = square(choice[n] - p_choice[n]); // for Brier score
        log_lik[n] = bernoulli_lpmf(choice[n] | p_choice[n]);
        log_lik_sess[k] += log_lik[n];
        brier_score_sess[k] += sq_err;
        n_obs_sess[k] += 1;
        brier_score += sq_err;
        total_obs += 1;
      }
    }
    brier_score_sess[k] /= n_obs_sess[k];
  }

  brier_score /= total_obs;
}