import os
from cmdstanpy.utils import cmdstan_version_at, cmdstan_path
cmdstan_path()
cmdstan_version_at(2,22)
from cmdstanpy.model import CmdStanModel
from cmdstanpy import _TMPDIR
_TMPDIR


# +

bernoulli_stan = os.path.join(cmdstan_path(), 'examples', 'bernoulli', 'bernoulli.stan')
bernoulli_model = CmdStanModel(stan_file=bernoulli_stan)
print(bernoulli_model)
print(bernoulli_model.code())
# -

bernoulli_json = os.path.join(cmdstan_path(), 'examples', 'bernoulli', 'bernoulli.data.json')
with open(bernoulli_json, 'r') as f:
    print(f.read())

bern_fit = bernoulli_model.sample(data=bernoulli_json, output_dir='.') 

print(bern_fit)

sampler_variables = bern_fit.sampler_vars_cols
stan_variables = bern_fit.stan_vars_cols
print('Sampler variables:\n{}'.format(sampler_variables)) 
print('Stan variables:\n{}'.format(stan_variables)) 


bern_fit.draws().shape

draws_theta = bern_fit.stan_variable(name='theta')
draws_theta.shape

bern_fit.summary()

bern_fit.sampler_diagnostics()





test_path = os.path.join('test','data')
logistic_stan = os.path.join(test_path,'logistic.stan')
logistic_data = os.path.join(test_path,'logistic.data.R')


logistic = CmdStanModel(stan_file=logistic_stan)
logistic_17 = logistic.sample(data=logistic_data, seed=12345, sig_figs=17)
smry = logistic_17.summary()
print(smry)
smry = logistic_17.summary(sig_figs=2)
print(smry)
smry = logistic_17.summary(sig_figs=17)
print(smry)

import numpy as np
np.set_printoptions(formatter={"float_kind": lambda x: "%g" % x})

print(smry)
