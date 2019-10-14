data { 
  int<lower=0> N; 
  real<lower=0,upper=1> theta;
} 
generated quantities {
  int y_sim[N];
  real<lower=0,upper=1> theta_rep;
  for (n in 1:N)
    y_sim[n] = bernoulli_rng(theta);
  theta_rep = sum(y) / N;
}
