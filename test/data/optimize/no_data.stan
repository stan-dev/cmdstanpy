parameters {
  real a;
}
model {
  target += normal_lupdf(a | 0, 1);
}

