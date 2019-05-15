#!/usr/bin/env bash

# install latest cmdstan release into subdir "releases"
# symlink release dir as "cmdstan"
# build binaries, compile example model to build model header

if [[ ! -e releases ]]; then
   mkdir releases
fi
if [[ ! -d releases ]]; then
    echo 'cannot install cmdstan, file "releases" is not a directory'
    exit 1
fi
pushd releases

echo "release dir: `pwd`"

TAG=`curl -s https://api.github.com/repos/stan-dev/cmdstan/releases/latest | grep "tag_name"`
echo $TAG > tmp-tag
VER=`perl -p -e 's/"tag_name": "v//g; s/",//g' tmp-tag`
echo $VER
rm tmp-tag

echo "latest cmdstan: $VER"
CS=cmdstan-${VER}
if [[ -d $cs && -f ${CS}/bin/stanc && -f ${CS}/examples/bernoulli/bernoulli ]]; then
    echo "cmdstan already installed"
#    exit 0
fi

curl -s -OL https://github.com/stan-dev/cmdstan/releases/download/v${VER}/${CS}.tar.gz
echo "download rc code: $?"
tar xzf ${CS}.tar.gz
echo "tar rc $?"

if [[ -h cmdstan ]]; then
    unlink cmdstan
fi
ln -s ${CS} cmdstan
cd cmdstan
make -j2 build examples/bernoulli/bernoulli
echo "installed ${CS}"
echo `ls -lFd releases/*`
