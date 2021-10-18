#!/usr/bin/env python
# Python code from Jupyter notebook `cmdstanpy_tutorial.ipynb`

# ### Import CmdStanPy classes and methods

import os.path
import sys

from cmdstanpy import CmdStanModel, cmdstan_path

# ### Instantiate & compile the model

bernoulli_dir = os.path.join(cmdstan_path(), 'examples', 'bernoulli')
stan_file = os.path.join(bernoulli_dir, 'bernoulli.stan')
with open(stan_file, 'r') as f:
    print(f.read())

model = CmdStanModel(stan_file=stan_file)
print(model)

# ### Assemble the data

data = {"N": 10, "y": [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]}

# In the CmdStan `examples/bernoulli` directory, there are data files in both `JSON` and `rdump` formats.
# bern_json = os.path.join(bernoulli_dir, 'bernoulli.data.json')

# ### Do Inference

fit = model.sample(data=data)
print(fit)

# ### Access the sample: the `CmdStanMCMC` object attributes and methods

print(fit.draws().shape)

# #### Get HMC sampler tuning parameters

print(fit.step_size)
print(fit.metric_type)
print(fit.metric)

# #### Summarize the results

print(fit.summary())

# #### Run sampler diagnostics

print(fit.diagnose())
