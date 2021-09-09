parameters {
  real z;
}
model {
  z ~ normal(0,1);
}
generated quantities {
  int x = 1;
  real a = 1;
  real b[2];
  real c[2,3];
  vector[2] d;
  vector[3] e[2];
  matrix[2,3] f[4];
  matrix[2,3] g[4,5];
  for (n in 1:2) {
    b[n] = 1;
    d[n] = 1;
    for (m in 1:3) {
      c[n,m] = n;
      e[n,m] = n;
      for (k in 1:4) {
        f[k, n, m] = n;
        for (j in 1:4) {
          g[k, j, n, m] = n;
        }
      }
    }
  }
}
