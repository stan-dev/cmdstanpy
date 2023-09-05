functions {
#include divide_real_by_two.stan
}
data {
  int<lower=0> N;
  array[N] int<lower=0,upper=1> y;
}
parameters {
  real<lower=0,upper=1> theta;
}
model {
  theta ~ beta(divide_real_by_two(2),1);
  for (n in 1:N)
    y[n] ~ bernoulli(theta);
}
