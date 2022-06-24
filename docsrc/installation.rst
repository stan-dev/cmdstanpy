Installation
============

CmdStanPy is a pure-Python3 package which wraps CmdStan,
the command-line interface to Stan which is written in C++.
Therefore, in addition to Python3,
CmdStanPy requires a modern C++ toolchain in order to build and run Stan models.
There are several ways to install CmdStanPy and the underlying CmdStan components.

* You can download CmdStanPy, CmdStan, and the C++ toolchain from conda-forge.

* You can download the CmdStanPy package from `PyPI <https://pypi.org>`_
  using `pip <https://pip.pypa.io/en/stable/>`_.
  
* If you want the current development version, you can clone the
  GitHub `CmdStanPy <https://github.com/stan-dev/cmdstanpy>`_ repository.

If you install CmdStanPy from PyPI or GitHub you will need to
install CmdStan as well, see section :ref:`CmdStan Installation <cmdstan-install>` below.

  
Conda: install CmdStanPy, CmdStan, C++ toolchain
------------------------------------------------

If you use `conda <https://docs.conda.io/en/latest/>`_,
you can install CmdStanPy and the underlying CmdStan components from the
`conda-forge <https://conda-forge.org/>`_ repository
via the following command:


.. code-block:: bash

    conda create -n stan -c conda-forge cmdstanpy


This command creates a new conda environment named ``stan`` and 
downloads and installs the ``cmdstanpy`` package
as well as CmdStan and the required C++ toolchain.

To install into an existing environment, use the conda ``install`` command instead of ``create``:


.. code-block:: bash

    conda install -c conda-forge cmdstanpy

Whichever installation method you use, afterwards you must
activate the new environment or deactivate/activate the existing one.
For example, if you installed cmdstanpy into a new environment ``stan``,
run the command

.. code-block:: bash

    conda activate stan


By default, the latest release of CmdStan is installed.
If you require a specific release of CmdStan, CmdStan versions
2.26.1 and *newer* can be installed by specifying
``cmdstan==VERSION`` in the install command.
Versions before 2.26.1 are not available from conda
but can be downloaded from the CmdStan
`releases <https://github.com/stan-dev/cmdstan/releases>`_ page.

A Conda environment is a directory that contains a specific collection of Conda packages.
To see the locations of your conda environments, use the command

.. code-block:: bash

    conda info -e

The shell environment variable ``CONDA_PREFIX`` points to the active conda environment (if any).
Both CmdStan and the C++ toolchain are installed into the
``bin`` subdirectory of the conda environment directory, i.e.,
``$CONDA_PREFIX/bin/cmdstan`` (Linux, MacOS), ``%CONDA_PREFIX%\bin\cmdstan`` (Windows).



PyPI: install package CmdStanPy
-------------------------------


CmdStan can also be installed from PyPI via URL: https://pypi.org/project/cmdstanpy/ or from the
command line using ``pip``:

.. code-block:: bash

    pip install --upgrade cmdstanpy

The optional packages are

* ``xarray``, an n-dimension labeled dataset package which can be used for outputs

To install CmdStanPy with all the optional packages:

.. code-block:: bash

    pip install --upgrade cmdstanpy[all]



GitHub: install from the CmdStanPy repository
---------------------------------------------

    
To install the current develop branch from GitHub:

.. code-block:: bash

    pip install -e git+https://github.com/stan-dev/cmdstanpy@/develop#egg=cmdstanpy


.. note::

  **Note for PyStan & RTools users:**  PyStan and CmdStanPy should be installed in
  separate environments if you are using the RTools toolchain (primarily Windows users).
  If you already have PyStan installed, you should take care to install CmdStanPy in its own
  virtual environment.


  **Jupyter notebook users:**  If you intend to run CmdStanPy from within a Jupyter notebook,
  you may need to install the
  `ipywidgets <https://ipywidgets.readthedocs.io/en/latest/index.html>`_.
  This will allow for progress bars implemented using the `tqdm <https://pypi.org/project/tqdm/>`_
  to display properly in the browser.
  For further help on Jupyter notebook installation and configuration , see
  `ipywidgets installation instructions <https://ipywidgets.readthedocs.io/en/latest/user_install.html#>`_
  and `this tqdm GitHub issue <https://github.com/tqdm/tqdm/issues/394#issuecomment-384743637>`_.


.. _cmdstan-install:

CmdStan Installation
--------------------

If you have installed CmdStanPy from PyPI or Github,
**you must install CmdStan**.
The recommended way to do so is via the ``install_cmdstan`` function
:ref:`described below<install-cmdstan-fun>`.

If you installed CmdStanPy with conda, CmdStan and the C++ toolchain,
both CmdStan and the C++ toolchain are installed into directory ``$CONDA_PREFIX/bin``
and you don't need to do any further installs.


C++ Toolchain Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^

To compile a Stan program requires a modern C++ compiler and the GNU-Make build utility.
These vary by operating system.

**Linux** The required C++ compiler is ``g++ 4.9 3``.
On most systems the GNU-Make utility is pre-installed and is the default ``make`` utility.
There is usually a pre-installed C++ compiler as well, but not necessarily new enough.

**MacOS** The Xcode and Xcode command line tools must be installed.  Xcode is available for free from the Mac App Store.
To install the Xcode command line tools, run the shell command: ``xcode-select --install``.

**Windows**  We recommend using the `RTools 4.0 <https://cran.r-project.org/bin/windows/Rtools/rtools40.html>`_ toolchain
which contains a ``g++ 8`` compiler and ``Mingw``, the native Windows equivalent of the GNU-Make utility.
This can be installed allong with CmdStan when you invoke the function :meth:`cmdstanpy.install_cmdstan`
with argument ``compiler=True``.


.. _install-cmdstan-fun:

Function ``install_cmdstan``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CmdStanPy provides the function :meth:`cmdstanpy.install_cmdstan` which
downloads CmdStan from GitHub and builds the CmdStan utilities.
It can be can be called from within Python or from the command line.
The default install location is a hidden directory in the user ``$HOME`` directory
named ``.cmdstan``.  This directory will be created by the install script.

+ From Python

.. code-block:: python

    import cmdstanpy
    cmdstanpy.install_cmdstan()
    cmdstanpy.install_cmdstan(compiler=True)  # only valid on Windows

+ From the command line on Linux or MacOSX

.. code-block:: bash

    install_cmdstan
    ls -F ~/.cmdstan

+ On Windows

.. code-block:: bash

    install_cmdstan --compiler True
    dir "%HOME%/.cmdstan"

The named arguments: ``-d <directory>`` and  ``-v <version>``
can be used to override these defaults:

.. code-block:: bash

    install_cmdstan -d my_local_cmdstan -v 2.27.0
    ls -F my_local_cmdstan

DIY Installation
^^^^^^^^^^^^^^^^

If you with to install CmdStan yourself, follow the instructions
in the `CmdStan User's Guide <https://mc-stan.org/docs/cmdstan-guide/cmdstan-installation.html>`_.

Locating the CmdStan installation directory
-------------------------------------------

CmdStanPy uses the environment variable ``CMDSTAN`` to register the CmdStan installation location.

+ If you use conda to install CmdStanPy, CmdStan is installed into location
  ``$CONDA_PREFIX/bin/cmdstan`` (Linux, MacOS), ``%CONDA_PREFIX%\bin\cmdstan`` (Windows)
  and the environment variable ``CMDSTAN`` is set accordingly.

+ If no environment variable ``CMDSTAN`` is set, CmdStanPy will try to locate
  a CmdStan installation in the default install location, which is a 
  directory named ``.cmdstan`` in your ``$HOME`` directory.

If you have installed CmdStan from a GitHub release or by cloning the CmdStan repository,
you will need to set this location, either via the ``CMDSTAN`` environment variable,
or via the CmdStanPy command ``set_cmdstan_path``

.. code-block:: python

    from cmdstanpy import cmdstan_path, set_cmdstan_path

    set_cmdstan_path(os.path.join('path','to','cmdstan'))
    cmdstan_path()
