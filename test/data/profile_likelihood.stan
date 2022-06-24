parameters {
  real x;
}
model {
  profile("likelihood") {
    x ~ normal(0, 1);
  }
}

