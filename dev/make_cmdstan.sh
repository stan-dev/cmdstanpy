#!/usr/bin/env bash

CSVER=${CSVER:-"2.18.1"}

if [[ -d cmdstan && -f cmdstan/bin/stanc ]]; then
    echo "cmdstan folder exists, skipping"
    exit 0
fi

set -eux
ls -lha
which g++
which clang++

curl -OL https://github.com/stan-dev/cmdstan/releases/download/v$CSVER/cmdstan-$CSVER.tar.gz
tar xzf cmdstan-$CSVER.tar.gz
mv cmdstan-$CSVER cmdstan
cd cmdstan
make -j2 build examples/bernoulli/bernoulli