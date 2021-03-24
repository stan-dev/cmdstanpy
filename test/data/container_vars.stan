generated quantities {
  int v_int = 1;
  real v_real = 2;
  array[2, 3] real v_2d_arr;
  for (i in 1:2)
    for (j in 1:3)
      v_2d_arr[i,j] = i*10 + j;
  matrix[2, 3] v_matrix;
  for (i in 1:2)
    for (j in 1:3)
      v_matrix[i,j] = i*10 + j;
}
