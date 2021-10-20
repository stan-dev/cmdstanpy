Installation
============

CmdStanPy is a pure-Python3 package, but it relies on CmdStan for all
of its functionality. There are several ways to install CmdStan and CmdStanPy, 
which depend on the kind of user you are. 


Conda users (Recommended)
-------------------------

If you use `conda <https://docs.conda.io/en/latest/>`__, 
installation of both can be done very simply. CmdStanPy
and CmdStan are both available via the 
`conda-forge <https://conda-forge.org/>`__ repository.

We recommend creating a new environment for CmdStan[Py]:

.. code-block:: bash

    conda create -n cmdstan -c conda-forge cmdstanpy

but installation is possible in an existing environment:

.. code-block:: bash

    conda install -c conda-forge cmdstanpy

These commands will install CmdStanPy, CmdStan, and the
required compilers for using CmdStan on your system inside
a conda environment. To use them, run ``conda activate cmdstan``,
or whichever name you used for your environment (following ``-n``
above).

Note that CmdStan is only available on conda for versions
2.27.0 and newer. If you require an older version, you must use
one of the following methods to install it. If you require a
version of CmdStan *newer* than 2.27.0, but not the latest,
you can install it in the standard conda way by specifying
``cmdstan==VERSION`` in the install command.

Pip (non-Conda) users 
-------------------------

CmdStan can also be installed from PyPI via URL: https://pypi.org/project/cmdstanpy/ or from the
command line using ``pip``:

.. code-block:: bash

    pip install --upgrade cmdstanpy

The optional packages are

* ``xarray``, an n-dimension labeled dataset package which can be used for outputs

To install CmdStanPy with all the optional packages:

.. code-block:: bash

    pip install --upgrade cmdstanpy[all]

To install the current develop branch from GitHub:

.. code-block:: bash

    pip install -e git+https://github.com/stan-dev/cmdstanpy@/develop#egg=cmdstanpy


If you install CmdStanPy from GitHub,
**you must install CmdStan**. The recommended way for Pip users 
to do so is via the ``install_cmdstan`` function 
:ref:`described below<install-cmdstan-fun>`

.. note::     

  **Note for PyStan & RTools users:**  PyStan and CmdStanPy should be installed in 
  separate environments if you are using the RTools toolchain (primarily Windows users).
  If you already have PyStan installed, you should take care to install CmdStanPy in its own
  virtual environment.


  **Jupyter notebook users:**  CmdStanPy can display progress bars during sampling
  as well as during the CmdStan build process; these are implemented using the
  `tqdm <https://github.com/tqdm/tqdm>`_ package,  which uses the
  `ipywidgets <https://ipywidgets.readthedocs.io/en/latest/index.html>`_ package
  in order to update the browser display.  For further help on installation
  and configuration, see
  `ipywidgets installation instructions <https://ipywidgets.readthedocs.io/en/latest/user_install.html#>`_
  and `this tqdm GitHub issue <https://github.com/tqdm/tqdm/issues/394#issuecomment-384743637>`_.

    

Installing CmdStan
------------------

Prerequisites
^^^^^^^^^^^^^

CmdStanPy requires an installed C++ toolchain
consisting of a modern C++ compiler and the GNU-Make utility.

+ Windows: CmdStanPy provides the function ``install_cxx_toolchain``

+ Linux: install g++ 4.9.3 or clang 6.0.  (GNU-Make is the default ``make`` utility)

+ maxOS:  install XCode and Xcode command line tools via command: `xcode-select --install`.

.. _install-cmdstan-fun:

Function ``install_cmdstan``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CmdStanPy provides the function :func:`~cmdstanpy.install_cmdstan` which
downloads CmdStan from GitHub and builds the CmdStan utilities.
It can be can be called from within Python or from the command line.

The default install location is a hidden directory in the user ``$HOME`` directory
named ``.cmdstan``.  This directory will be created by the install script.

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
is a directory named ``.cmdstan`` in your ``$HOME`` directory. [1]_
If you have installed CmdStan in a different directory,
then you can set the environment variable ``CMDSTAN`` to this
location and it will be picked up by CmdStanPy. `Note:` This is done
for you if you installed via ``conda``, as cmdstan will be installed
in the ``bin/`` subfolder of the environment directory.

.. code-block:: bash

    export CMDSTAN='/path/to/cmdstan-2.27.0'


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


.. rubric:: Footnotes

.. [1]  In very early versions, this hidden directory was named ``.cmdstanpy``

    
