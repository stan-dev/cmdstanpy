data {
  int<lower=0> N;
  array[N] int<lower=0, upper=1> y;
}
parameters {
  real<lower=0, upper=1> theta;
}
model {
  theta ~ beta(1, 1);
  y ~ bernoulli(theta);
}
generated quantities {
  array[N] int y_rep;
  for (n in 1 : N) {
    // deterministic value
    y_rep[n] = y[n];
  }
}

