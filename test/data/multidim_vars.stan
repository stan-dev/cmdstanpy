// run with data file logistic.data.R
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
generated quantities {
  array[5, 4, 3] int<lower=0, upper=1> y_rep;
  int<lower=0, upper=60> frac_60 = 0; // 5*4*3 = 60
  for (i in 1 : 5) {
    for (j in 1 : 4) {
      for (k in 1 : 3) {
        y_rep[i, j, k] = bernoulli_logit_rng(x[min(i + j + k, N)] * beta);
        if (y_rep[i, j, k] == 1) {
          frac_60 += 1;
        }
      }
    }
  }
}

