FROM python:3.6-stretch

ENV CSVER=2.17.1
ENV CMDSTAN=/opt/cmdstan-$CSVER
ENV CXX=clang++-3.9
ENV MPLBACKEND=agg
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y clang-3.9
RUN pip install numpy coverage pytest pytest-cov pytest-xdist filelock sphinx
RUN pip install --upgrade setuptools wheel twine

WORKDIR /opt/
RUN curl -OL https://github.com/stan-dev/cmdstan/releases/download/v$CSVER/cmdstan-$CSVER.tar.gz \
 && tar xzf cmdstan-$CSVER.tar.gz \
 && rm -rf cmdstan-$CSVER.tar.gz \
 && cd cmdstan-$CSVER \
 && make -j2 build examples/bernoulli/bernoulli

RUN mkdir -p /opt/pycmdstan
WORKDIR /opt/pycmdstan
RUN pip install scipy
ADD ./ /opt/pycmdstan/
