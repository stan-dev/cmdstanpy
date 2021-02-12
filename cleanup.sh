#!/bin/sh

if [[ -e cmdstanpy/__pycache__ ]]; then
  echo "remove cmdstanpy/__pycache__ dir"
  rm -rf cmdstanpy/__pycache__
fi

if [[ -e test/__pycache__ ]]; then
  echo "remove test/__pycache__ dir"
  rm -rf test/__pycache__
fi

if [[ -e docs/_build ]]; then
  echo "remove docs/_build dir"
  rm -rf docs/_build
fi

echo "remove compiled stan programs"
find test/data -perm 755 -and -type f
find test/data -perm 755 -and -type f -exec rm {} \;
find docs/notebooks -perm 755 -and -type f
find docs/notebooks -perm 755 -and -type f -exec rm {} \;

echo "remove stan program .hpp files"
find test/data -name "*.hpp"
find test/data -name "*.hpp" -exec rm {} \;
find docs/notebooks -name "*.hpp"
find docs/notebooks -name "*.hpp" -exec rm {} \;

echo "remove stan program .d files"
find test/data -name "*.d"
find test/data -name "*.d" -exec rm {} \;
find docs/notebooks -name "*.d"
find docs/notebooks -name "*.d" -exec rm {} \;

echo "remove stan program .o files"
find test/data -name  "*.o"
find test/data -name  "*.o" -exec rm {} \;
find docs/notebooks -name  "*.o"
find docs/notebooks -name  "*.o" -exec rm {} \;
