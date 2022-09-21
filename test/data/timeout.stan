data {
    // Indicator for endless looping.
    int loop;
}

transformed data {
    // Maybe loop forever so the model times out.
    real y = 1;
    while(loop && y) {
        y += 1;
    }
}

parameters {
    real x;
}

model {
    // A nice model so we can get a fit for the `generated_quantities` call.
    x ~ normal(0, 1);
}
