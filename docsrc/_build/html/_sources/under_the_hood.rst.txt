Under the Hood
______________

Under the hood, CmdStanPy uses the CmdStan command line interface
to compile and fit a model to data.
The function ``cmdstan_path`` returns the path to the local CmdStan installation.
See the installation section for more details on installing CmdStan.


File Handling
-------------

CmdStan is file-based interface, therefore CmdStanPy
maintains the necessary files for all models, data, and
inference method results.
CmdStanPy uses the Python library ``tempfile`` module to create
a temporary directory where all input and output files are written and
which is deleted when the Python session is terminated.


Input Data
^^^^^^^^^^

When the input data for the ``CmdStanModel`` inference methods
is supplied as a Python dictionary, this data is written to disk as
the corresponding JSON object.

Output Files
^^^^^^^^^^^^

Output filenames are composed of the model name, a timestamp
in the form 'YYYYMMDDhhmm' and the chain id, plus the corresponding
filetype suffix, either '.csv' for the CmdStan output or '.txt' for
the console messages, e.g. `bernoulli-201912081451-1.csv`. Output files
written to the temporary directory contain an additional 8-character
random string, e.g. `bernoulli-201912081451-1-5nm6as7u.csv`.


When the ``output_dir`` argument to the ``CmdStanModel`` inference methods
is given, output files are written to the specified directory, otherwise
they are written to the session-specific output directory.
All fitted model objects, i.e. ``CmdStanMCMC``, ``CmdStanVB``, ``CmdStanMLE``,
and ``CmdStanGQ``, have method ``save_csvfiles`` which moves the output files
to a specified directory.
