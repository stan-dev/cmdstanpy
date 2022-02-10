generated quantities {
  int a  = 1;
  array[2,3,2] int ys = {{{3,0},{0,4}, {5,0}}, {{0,1}, {0,2}, {0,3}}};
  array[2,3] complex zs = {{3,4i,5},{1i,2i,3i}};
  print(ys);
  print(zs);
}
