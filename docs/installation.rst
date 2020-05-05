Installation
____________

Install package CmdStanPy
-------------------------

CmdStanPy is a pure-Python package which can be installed from PyPI

.. code-block:: bash

    pip install --upgrade cmdstanpy

or from GitHub

.. code-block:: bash

    pip install -e git+https://github.com/stan-dev/cmdstanpy#egg=cmdstanpy

To install CmdStanPy with all the optional packages
(ujson; json processing, tqdm; progress bar)

.. code-block:: bash

    pip install --upgrade cmdstanpy[all]

*Note for PyStan users:*  PyStan and CmdStanPy should be installed in separate environments.
If you already have PyStan installed, you should take care to install CmdStanPy in its own
virtual environment.

User can install optional packages with pip with the CmdStanPy installation

.. code-block:: bash

    pip install --upgrade cmdstanpy[all]


The optional packages are

  * ``ujson`` which provides faster IO
  * ``tqdm`` which displays a progress during sampling

To install these manually

.. code-block:: bash

    pip install ujson
    pip install tqdm


Install CmdStan
---------------

CmdStanPy requires a local install of CmdStan.

Prerequisites
^^^^^^^^^^^^^

CmdStanPy requires an installed C++ toolchain.

Function ``install_cmdstan``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CmdStanPy provides the function ``install_cmdstan`` which
downloads CmdStan from GitHub and builds the CmdStan utilities.
It can be can be called from within Python or from the command line.
By default it installs the latest version of CmdStan into a directory named
``.cmdstanpy`` in your ``$HOME`` directory:

+ From Python

.. code-block:: python

    import cmdstanpy
    cmdstanpy.install_cmdstan()

+ From the command line on Linux or MacOSX

.. code-block:: bash

    install_cmdstan
    ls -F ~/.cmdstanpy

+ On Windows

.. code-block:: bash

    python -m cmdstanpy.install_cmdstan
    dir "%HOME%/.cmdstanpy"

The named arguments: `-d <directory>` and  `-v <version>`
can be used to override these defaults:

.. code-block:: bash

    install_cmdstan -d my_local_cmdstan -v 2.20.0
    ls -F my_local_cmdstan


Specifying CmdStan installation location
""""""""""""""""""""""""""""""""""""""""

The default for the CmdStan installation location
is a directory named ``.cmdstanpy`` in your ``$HOME`` directory.

If you have installed CmdStan in a different directory,
then you can set the environment variable ``CMDSTAN`` to this
location and it will be picked up by CmdStanPy:

.. code-block:: bash

    export CMDSTAN='/path/to/cmdstan-2.20.0'


The CmdStanPy commands ``cmdstan_path`` and ``set_cmdstan_path``
get and set this environment variable:

.. code-block:: python

    from cmdstanpy import cmdstan_path, set_cmdstan_path

    oldpath = cmdstan_path()
    set_cmdstan_path(os.path.join('path','to','cmdstan'))
    newpath = cmdstan_path()


Specifying a custom ``make`` tool
"""""""""""""""""""""""""""""""""

To use custom ``make``-tool use ``set_make_env`` function.

.. code-block:: python

    from cmdstanpy import set_make_env
    set_make_env("mingw32-make.exe") # On Windows with mingw32-make



    

