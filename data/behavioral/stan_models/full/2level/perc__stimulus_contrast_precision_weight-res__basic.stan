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

            // Prediction (level 1)
            p_hat = inv_logit(mu2_hat);
            var1_hat = p_hat * (1.0 - p_hat);
            m[n] = p_hat;

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
    vector m, 
    array[] int start_idx,
    array[] int end_idx,
    vector zeta, 
    vector lapse
  ) {
    int N = num_elements(m);
    int K = size(start_idx);
    vector[N] p_choice;
    real theta;

    for (k in 1:K) {
      for (n in start_idx[k]:end_idx[k]) {
        real m_safe = fmin(1 - 1e-12, fmax(1e-12, m[n]));
        theta = zeta[k] * logit(m_safe);
        p_choice[n] = (1.0 - lapse[k]) * inv_logit(theta) + 0.5 * lapse[k];
      }
    }

    return p_choice;
  }
}

data {
  int<lower=1> N_subj; // num subjects
  int<lower=1> N_obs; // num. total observations (stacked for all sessions)
  int<lower=1> N_sess; // num. sessions
  array[N_sess] int<lower=1, upper=N_subj> subj_ids;

  array[N_obs] int<lower=0, upper=1> stimulus_side;
  vector<lower=0, upper=1>[N_obs] stimulus_contrast;
  array[N_obs] int<lower=-1, upper=1> choice;

  array[N_sess] int<lower=1> start_idx;
  array[N_sess] int<lower=1> end_idx;
}

parameters {
  vector[N_subj] omega2_subj_eta;
  vector[N_subj] zeta_subj_eta;
  vector[N_subj] contrast_slope_subj_eta;
  vector[N_subj] contrast_midpoint_subj_eta;
  vector[N_subj] lapse_subj_eta;

  vector[N_sess] omega2_session_eta;
  vector[N_sess] zeta_session_eta;

  // OMEGA2: Population -> Subj -> Session 
  real mu_omega2_population;
  real<lower=0> sigma_omega2_population;
  real<lower=0> sigma_omega2_subj;

  // ZETA: Population -> Subj -> Session 
  real mu_zeta_population;
  real<lower=0> sigma_zeta_population;
  real<lower=0> sigma_zeta_subj;

  // CONTRAST_SLOPE: Population -> Subj 
  real mu_contrast_slope_population;
  real<lower=0> sigma_contrast_slope_population;

  // CONTRAST_MIDPOINT: Population -> Subj 
  real mu_contrast_midpoint_population;
  real<lower=0> sigma_contrast_midpoint_population;

  // LAPSE: Population -> Subj 
  real mu_lapse_population;
  real<lower=0> sigma_lapse_population;
}

transformed parameters {
  vector[N_subj] mus_omega2_subj;
  vector<upper=0>[N_sess] omega2_session;

  vector[N_subj] mus_zeta_subj;
  vector<lower=0>[N_sess] zeta_session;

  vector[N_subj] contrast_slope_subj_latent;
  vector<lower=0.5, upper=5>[N_subj] contrast_slope_subj;

  vector[N_subj] contrast_midpoint_subj_latent;
  vector<lower=0, upper=1>[N_subj] contrast_midpoint_subj;

  vector[N_subj] lapse_subj_latent;
  vector<lower=0.001, upper=0.2>[N_subj] lapse_subj;

  vector[N_obs] x_1_expected_mean;
  vector[N_obs] p_choice;

  mus_omega2_subj = mu_omega2_population + sigma_omega2_population * omega2_subj_eta;
  omega2_session = -log1p_exp(mus_omega2_subj[subj_ids] + sigma_omega2_subj * omega2_session_eta);

  mus_zeta_subj = mu_zeta_population + sigma_zeta_population * zeta_subj_eta;
  zeta_session = exp(mus_zeta_subj[subj_ids] + sigma_zeta_subj * zeta_session_eta);

  contrast_slope_subj_latent = mu_contrast_slope_population + sigma_contrast_slope_population * contrast_slope_subj_eta;
  contrast_slope_subj = 0.5 + (5 - 0.5) * inv_logit(contrast_slope_subj_latent);

  contrast_midpoint_subj_latent = mu_contrast_midpoint_population + sigma_contrast_midpoint_population * contrast_midpoint_subj_eta;
  contrast_midpoint_subj = inv_logit(contrast_midpoint_subj_latent);

  lapse_subj_latent = mu_lapse_population + sigma_lapse_population * lapse_subj_eta;
  lapse_subj = 0.001 + (0.2 - 0.001) * inv_logit(lapse_subj_latent);

  // Perceptual -> Response Probs
  x_1_expected_mean = perceptual_model(
    stimulus_side, stimulus_contrast, 
    start_idx, end_idx,
    omega2_session, 
    contrast_slope_subj[subj_ids], 
    contrast_midpoint_subj[subj_ids]
  );

  p_choice = response_model(
    x_1_expected_mean, 
    start_idx, end_idx,
    zeta_session, 
    lapse_subj[subj_ids]
  );
}

model {

  // Population-Level Prior
  target += std_normal_lpdf(mu_omega2_population);
  target += std_normal_lpdf(sigma_omega2_population);
  target += std_normal_lpdf(sigma_omega2_subj);

  target += std_normal_lpdf(mu_zeta_population);
  target += std_normal_lpdf(sigma_zeta_population);
  target += std_normal_lpdf(sigma_zeta_subj);

  target += std_normal_lpdf(mu_contrast_slope_population);
  target += std_normal_lpdf(sigma_contrast_slope_population);

  target += std_normal_lpdf(mu_contrast_midpoint_population);
  target += std_normal_lpdf(sigma_contrast_midpoint_population);

  target += std_normal_lpdf(mu_lapse_population);
  target += std_normal_lpdf(sigma_lapse_population);

  // Subject-Level Prior
  target += std_normal_lpdf(omega2_subj_eta);
  target += std_normal_lpdf(zeta_subj_eta);
  target += std_normal_lpdf(contrast_slope_subj_eta);
  target += std_normal_lpdf(contrast_midpoint_subj_eta);
  target += std_normal_lpdf(lapse_subj_eta);

  // Session-Level Prior
  target += std_normal_lpdf(omega2_session_eta);
  target += std_normal_lpdf(zeta_session_eta);

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