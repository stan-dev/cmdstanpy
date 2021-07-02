#!/usr/bin/env python
# Python code from Jupyter notebook `cmdstanpy_tutorial.ipynb`

# ### Import CmdStanPy classes and methods

import matplotlib
import pandas as pd
import os

from cmdstanpy import cmdstan_path, CmdStanModel

# ### Instantiate & compile the model

bernoulli_dir = os.path.join(cmdstan_path(), 'examples', 'bernoulli')
bernoulli_stan = os.path.join(bernoulli_dir, 'bernoulli.stan')
with open(bernoulli_stan, 'r') as f:
    print(f.read())

bernoulli_model = CmdStanModel(stan_file=bernoulli_stan)
print(bernoulli_model)

# ### Assemble the data

bern_data = {"N": 10, "y": [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]}

# In the CmdStan `examples/bernoulli` directory, there are data files in both `JSON` and `rdump` formats.
# bern_json = os.path.join(bernoulli_dir, 'bernoulli.data.json')

# ### Do Inference

bern_fit = bernoulli_model.sample(data=bern_data)
print(bern_fit)

# ### Access the sample: the `CmdStanMCMC` object attributes and methods

bern_fit.draws().shape

vars = bern_fit.stan_variables()
for (k, v) in vars.items():
    print(k, v.shape)

thetas = bern_fit.stan_variable(name='theta')
pd.DataFrame(data=thetas).plot.density()

# #### Get HMC sampler tuning parameters

bern_fit.stepsize
bern_fit.metric_type
bern_fit.metric

# #### Summarize the results

bern_fit.summary()

# #### Run sampler diagnostics

bern_fit.diagnose()
