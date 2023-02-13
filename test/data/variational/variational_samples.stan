data {
    int n;
    vector[n] vec;
    complex_matrix[n, 2 * n] cmat;
    real scalar;
    array [n, 2] vector[5] arrvec;
}

parameters {
    real theta;
}

model {
    theta ~ normal(0, 1);
}

generated quantities {
    // Generated quantities are inputs with theta added.
    vector[n] gvec = theta + vec;
    complex_matrix[n, 2 * n] gcmat = theta + cmat;
    real gscalar = scalar + theta;
    array[n, 2] vector[5] garrvec;
    for (i in 1:n) {
        garrvec[i, 1] = arrvec[i, 1] + theta;
        garrvec[i, 2] = arrvec[i, 2] + theta;
    }
}
