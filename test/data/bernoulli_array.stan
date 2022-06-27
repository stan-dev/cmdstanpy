data {
  int<lower=0> N;
  array[N] int<lower=0, upper=1> y;
}
parameters {
  array[1] real<lower=0, upper=1> theta;
}
model {
  theta[1] ~ beta(1, 1); // uniform prior on interval 0,1
  y ~ bernoulli(theta[1]);
}

