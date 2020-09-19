
- ``data``: Values for all data variables in the model, specified either as a dictionary with entries matching the data variables, or as the path of a data file in JSON or Rdump format.

- ``seed``: The seed for random number generator.

- ``inits``:  Specifies how the sampler initializes parameter values.
            
- ``output_dir``:  Name of the directory to which CmdStan output files are written.

- ``save_diagnostics``:  Whether or not to the CmdStan auxiliary output file.
  For the ``sample`` method, the diagnostics file contains sampler information for each draw
  together with the gradients on the unconstrained scale and log probabilities for all parameters in the model.
