data {
  int<lower=0> N;
  int<lower=0,upper=1> y[N];
}
parameters {
  real<lower=0,upper=1> theta;
}
model {
  theta ~ beta(1,1);
  y ~ bernoulli(theta);
}
generated quantities {
  int y_rep[N];
  for (n in 1:N)
    // deterministic value
    y_rep[n] = y[n];
}
