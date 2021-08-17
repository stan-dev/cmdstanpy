data { 
  int<lower=0> N;
  real<lower=0,upper=1> theta;
} 
generated quantities {
  int theta_rep = 0;
  for (n in 1:N)
    theta_rep += bernoulli_rng(theta);
}