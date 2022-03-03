data {
  int<lower=0> N;
  int<lower=0,upper=1> y[N];
}
parameters {
  real<lower=0,upper=1> theta;
}
model {
  theta ~ beta(1,1);  // uniform prior on interval 0,1
  y ~ bernoulli(theta);
}

generated quantities {
   // these should be accessible via .
   real a = 4.5;
   array[3] real b = {1, 2.5, 4.5};

   // these should not override built in properties/funs
   real thin = 3.5;
   int draws = 0;
   int optimized_params_np = 0;
   int variational_params_np = 0;
}
