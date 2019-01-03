# CmdStanPy

CmdStanPy is a lightweight interface to Stan for Python users which
provides the necessary objects and functions to compile a Stan program
and run Stan's samplers.

### Goals

- Clean interface to Stan services so that CmdStanPy can keep up with Stan releases.

- Provides complete control - all sampler arguments have corresponding named argument
for CmdStanPy sampler function.

- Easy to install,
  + minimal Python library dependencies: numpy, pandas
  + Python code doesn't interface directly with c++, only calls compiled executables

- Modular - CmdStanPy produces a sample from the posterior, downstream modules do the analysis.


### Source Repository

CmdStan's source-code repository is hosted here on GitHub.

### Licensing

The CmdStanPy, CmdStan, and the core Stan C++ code are licensed under new BSD.
