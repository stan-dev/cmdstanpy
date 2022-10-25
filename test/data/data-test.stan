data {
  real inf;
  real nan;
}


generated quantities {
  print(inf);
  print(nan);
  real inf_out = inf;
  real nan_out = nan;
}
