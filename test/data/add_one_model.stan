functions {
    #include add_one_function.stan
}

generated quantities {
    real x = add_one(3);
}
