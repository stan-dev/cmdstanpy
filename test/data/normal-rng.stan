transformed data{
  real x_;
  x_ = std_normal_rng();
}
generated quantities{
  real x = x_;
}
