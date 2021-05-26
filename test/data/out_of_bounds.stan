// Stanc3 compiler through 2.26 segfaults
parameters {
  row_vector[10] x[10];
}
transformed parameters {
  print(x[200,200]);
}
model {
  for (i in 1:10) 
    x[i] ~ normal(0,1);
}
