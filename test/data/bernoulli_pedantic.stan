data {
  int<lower=0> N;
  array[N] int<lower=0, upper=1> y;
}
parameters {
  real<lower=0, upper=1> theta;
}
model {
  // no prior on theta - should produce a warning
  y ~ bernoulli(theta);
}

