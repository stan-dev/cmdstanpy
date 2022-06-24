// generate parameters, data, outcomes for poisson regression 
transformed data {
  int N_obs = 20; // specify constants
}
generated quantities {
  int N = N_obs;
  array[N_obs] int y_sim;
  vector[N_obs] x_sim;
  vector[N_obs] pop_sim;
  real alpha_sim;
  real beta_sim;
  array[N_obs] real eta; // poisson variate
  
  alpha_sim = normal_rng(0, 1); // simulate parameters
  beta_sim = normal_rng(0, 1); // from prior
  
  for (n in 1 : N_obs) {
    // simulate data from params
    pop_sim[n] = uniform_rng(500, 2500);
    x_sim[n] = uniform_rng(0.01, 0.99);
    eta[n] = log(pop_sim[n]) + alpha_sim + x_sim[n] * beta_sim;
    y_sim[n] = poisson_log_rng(eta[n]);
  }
}

