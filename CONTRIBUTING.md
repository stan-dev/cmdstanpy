We welcome contributions to the project and we could really use your help to:

* Investigate and fix reported bugs

* Improve the workflow

* Improve the documentation

* Increase test coverage


### Developers

We welcome contributions!  Contributions should follow the general outlines of the [Stan Developer Process](https://github.com/stan-dev/stan/wiki/Developer-process-overview)

* GitHub issues are used to discuss both bugs and features and propose implementations.

* Additions and changes to the code base should include updates to unit tests. User-facing changes require updates to the documentation as well.

* The GitHub repo organization follows the gitflow model described by Vincent Driessen in the blog post ["A successful Git branching model."](https://nvie.com/posts/a-successful-git-branching-model/).  The main developer branch is `develop`; it should always be runnable.

* Unit tests must be runnable under both the [unittest](https://docs.python.org/3/library/unittest.html#module-unittest) and [PyTest](https://docs.pytest.org/en/stable/) frameworks.

* Both [PyLint](https://www.pylint.org) and [Flake8](https://flake8.pycqa.org/en/latest/) are used to check code style and formatting according to rules in https://github.com/stan-dev/cmdstanpy/blob/develop/.pylintrc  and https://github.com/stan-dev/cmdstanpy/blob/a6e09190af555aa6d05993d630ebf39a3d4bb867/.travis.yml#L30. Code is formatted with [isort](https://pycqa.github.io/isort/) and [black](https://black.readthedocs.io/en/stable/).

* We recommend using [pre-commit](https://pre-commit.com/) to ensure the above code style and quality checks are run whenever you make code edits. You can run `pre-commit install` from this directory to install these hooks. Note: both pre-commit and pylint need to be installed before running this command.


### Documentation

CmdStanPy uses [Sphinx](https://www.sphinx-doc.org/en/master/) to generate the docs.

* Documentation src files live in directory `docsrc`

* The documentation is hosted on [readthedocs](https://readthedocs.org) as https://cmdstanpy.readthedocs.io, which provides documentation for all tagged releases.

   + The `stable` branch is the most recent tagged version

   + The `latest` branch is the current `develop` branch

   + [Readthedocs automation rules](https://docs.readthedocs.io/en/stable/automation-rules.html) are used to generate docs for new tags.

* The current `develop` branch docset in the `docs` directory is hosted by GitHub pages as https://mc-stan.org/cmdstanpy  The Sphinx makefile `docsrc/Makefile` target `github` is used to update the contents of the `docs` directory.
