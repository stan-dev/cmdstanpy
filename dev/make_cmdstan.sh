#!/usr/bin/env bash

CSVER=${CSVER:-"2.19.0"}
cs=cmdstan-$CSVER

if [[ -d $cs && -f $cs/examples/bernoulli/bernoulli ]]; then
    echo "cmdstan folder ($cs) exists, skipping"
    exit 0
fi

curl -OL https://github.com/stan-dev/cmdstan/releases/download/v$CSVER/cmdstan-$CSVER.tar.gz
tar xzf $cs.tar.gz
mv $cs cmdstan
cd cmdstan
make -j2 build examples/bernoulli/bernoulli
