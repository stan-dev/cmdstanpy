data {
  tuple(real, int, real) x;
  array[3] tuple(real, matrix[4, 5]) y;
}
transformed data {
  print("x: ", x);
  print("y: ", y[1]);
}
