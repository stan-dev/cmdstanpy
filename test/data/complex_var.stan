transformed data {
  array[10] int<lower=0, upper=1> y = {0, 1, 0, 0, 0, 0, 0, 0, 0, 1};
}
parameters {
  real<lower=0, upper=1> theta;
}
model {
  theta ~ beta(1, 1);
  y ~ bernoulli(theta);
}
// model segment is just so that VB works

generated quantities {
  int a = 1;
  array[2, 3, 2] int ys = {{{3, 0}, {0, 4}, {5, 0}},
                           {{0, 1}, {0, 2}, {0, 3}}};
  array[2, 3] complex zs = {{3, 4i, 5}, {1i, 2i, 3i}};
  complex z = 3 + 4i;
  
  array[2] int imag = {3, 4};
}

