Installation
============

Both CmdStanPy and CmdStan must be installed;
since the CmdStanPy package contains utility ``install_cmdstan``,
we recommend installing the CmdStanPy package first.


Install package CmdStanPy
-------------------------

CmdStanPy is a pure-Python3 package.

It can be installed from PyPI via URL: https://pypi.org/project/cmdstanpy/ or from the
command line using ``pip``:

.. code-block:: bash

    pip install --upgrade cmdstanpy

The optional packages are

  * ``tqdm`` which allows for progress bar display during sampling
  * ``ujson`` which provides faster IO

To install CmdStanPy with all the optional packages:

.. code-block:: bash

    pip install --upgrade cmdstanpy[all]

To install the current develop branch from GitHub:

.. code-block:: bash

    pip install -e git+https://github.com/stan-dev/cmdstanpy@/develop


*Note for PyStan users:*  PyStan and CmdStanPy should be installed in separate environments.
If you already have PyStan installed, you should take care to install CmdStanPy in its own
virtual environment.

Install CmdStan
---------------

Prerequisites
^^^^^^^^^^^^^

CmdStanPy requires an installed C++ toolchain
consisting of a modern C++ compiler and the GNU-Make utility.

+ Windows: CmdStanPy provides the function ``install_cxx_toolchain``

+ Linux: install g++ 4.9.3 or clang 6.0.  (GNU-Make is the default ``make`` utility)

+ maxOS:  install XCode and Xcode command line tools via command: `xcode-select --install`.

Function ``install_cmdstan``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CmdStanPy provides the function :ref:`function_install_cmdstan` which
downloads CmdStan from GitHub and builds the CmdStan utilities.
It can be can be called from within Python or from the command line.

The default install location is a hidden directory in the user ``$HOME`` directory
named ``.cmdstan``.  (In earlier versions, the hidden directory was named ``.cmdstanpy``,
and if directory ``$HOME/.cmdstanpy`` exists, it will continue to be used as the
default install dir.)  This directory will be created by the install script.

+ From Python

.. code-block:: python

    import cmdstanpy
    cmdstanpy.install_cmdstan()

+ From the command line on Linux or MacOSX

.. code-block:: bash

    install_cmdstan
    ls -F ~/.cmdstan

+ On Windows

.. code-block:: bash

    python -m cmdstanpy.install_cmdstan
    dir "%HOME%/.cmdstan"

The named arguments: `-d <directory>` and  `-v <version>`
can be used to override these defaults:

.. code-block:: bash

    install_cmdstan -d my_local_cmdstan -v 2.20.0
    ls -F my_local_cmdstan

DIY Installation 
^^^^^^^^^^^^^^^^

If you with to install CmdStan yourself, follow the instructions
in the `CmdStan User's Guide <https://mc-stan.org/docs/cmdstan-guide/cmdstan-installation.html>`__.

Post Installation: Setting Environment Variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default for the CmdStan installation location
is a directory named ``.cmdstan`` in your ``$HOME`` directory.
(In earlier versions, the hidden directory was named ``.cmdstanpy``,
and if directory ``$HOME/.cmdstanpy`` exists, it will continue to be used as the
default install dir.)

If you have installed CmdStan in a different directory,
then you can set the environment variable ``CMDSTAN`` to this
location and it will be picked up by CmdStanPy:

.. code-block:: bash

    export CMDSTAN='/path/to/cmdstan-2.24.0'


The CmdStanPy commands ``cmdstan_path`` and ``set_cmdstan_path``
get and set this environment variable:

.. code-block:: python

    from cmdstanpy import cmdstan_path, set_cmdstan_path

    oldpath = cmdstan_path()
    set_cmdstan_path(os.path.join('path','to','cmdstan'))
    newpath = cmdstan_path()

To use custom ``make``-tool use ``set_make_env`` function.

.. code-block:: python

    from cmdstanpy import set_make_env
    set_make_env("mingw32-make.exe") # On Windows with mingw32-make
