data {
  int<lower=0> N; // number of items
  int<lower=0> M; // number of predictors
  array[N] int<lower=0, upper=1> y; // outcomes
  array[N] row_vector[M] x; // predictors
}
parameters {
  vector[M] beta; // coefficients
}
model {
  for (m in 1 : M) {
    beta[m] ~ cauchy(0.0, 2.5);
  }
  
  for (n in 1 : N) {
    y[n] ~ bernoulli_logit(x[n] * beta);
  }
}

