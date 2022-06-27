data {
  int<lower=0> N;
  array[N] int<lower=0, upper=1> y;
}
parameters {
  real<lower=0, upper=1> theta;
}
model {
  theta ~ beta(1, 1)
  for (n in 1:N)
    y[n] ~ bernoulli(theta);
}
