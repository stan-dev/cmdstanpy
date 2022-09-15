.. py:currentmodule:: cmdstanpy

Controlling Outputs
===================

CSV File Outputs
----------------

Underlyingly, the CmdStan outputs are a set of per-chain
`Stan CSV files <https://mc-stan.org/docs/cmdstan-guide/stan-csv.html#mcmc-sampler-csv-output>`__.
The filenames follow the template '<model_name>-<YYYYMMDDHHMMSS>-<chain_id>'
plus the file suffix '.csv'.
CmdStanPy also captures the per-chain console and error messages.

.. ipython:: python

    import os
    from cmdstanpy import CmdStanModel
    stan_file = os.path.join('users-guide', 'examples', 'bernoulli.stan')
    model = CmdStanModel(stan_file=stan_file)

    data_file = os.path.join('users-guide', 'examples', 'bernoulli.data.json')
    fit = model.sample(data=data_file)

    # printing the object reports sampler commands, output files
    print(fit)

The ``output_dir`` argument is an optional argument which specifies
the path to the output directory used by CmdStan.
If this argument is omitted, the output files are written
to a temporary directory which is deleted when the current Python session is terminated.

.. ipython:: python

    fit = model.sample(data=data_file, output_dir="./outputs/")

    !ls outputs/

Alternatively, the :meth:`~CmdStanMCMC.save_csvfiles` function moves the CSV files
to a specified directory.

.. ipython:: python

    fit = model.sample(data=data_file)
    fit.save_csvfiles(dir='some/path')

    !ls some/path

.. ipython:: python
    :suppress:

    !rm -rf outputs/ some/path/

Logging
-------

You may notice CmdStanPy can produce a lot of output when it is running:

.. ipython:: python

    fit = model.sample(data=data_file, show_progress=False)

This output is managed through the built-in :mod:`logging` module. For example, it can be disabled entirely:

.. ipython:: python

    import logging
    cmdstanpy_logger = logging.getLogger("cmdstanpy")
    cmdstanpy_logger.disabled = True
    # look, no output!
    fit = model.sample(data=data_file, show_progress=False)

Or one can remove the logging handler that CmdStanPy installs by default and install their own for more
fine-grained control. For example, the following code sends all logs (including the ``DEBUG`` logs, which are hidden by default),
to a file.

DEBUG logging is useful primarily to developers or when trying to hunt down an issue.

.. ipython:: python

    cmdstanpy_logger.disabled = False
    # remove all existing handlers
    cmdstanpy_logger.handlers = []

    cmdstanpy_logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler('all.log')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            "%H:%M:%S",
        )
    )
    cmdstanpy_logger.addHandler(handler)

Now, if we run the model and check the contents of the file, we will see all the possible logging.

.. ipython:: python

    fit = model.sample(data=data_file, show_progress=False)

    with open('all.log','r') as logs:
        for line in logs.readlines():
            print(line.strip())

.. ipython:: python
    :suppress:

    !rm all.log
