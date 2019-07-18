from cmdstanpy.model import Model
from cmdstanpy.stanfit import StanFit
model_a = Model(stan_file='test/data/bernoulli.stan')
model_a.compile()
stanfit = model_a.sample(data='test/data/bernoulli.data.json')

model_b = Model(stan_file='test/data/bernoulli_ppc/bernoulli_ppc.stan')
x = model_b.compile()
output = "/Users/aparna.chaganty/adhoc-datascience/cmdstanpy/test/data/bernoulli_ppc/test_sampling_output"
bern_fit = model_b.sample(data='test/data/bernoulli.data.json', csv_basename=output)
model_b.run_generate_quantities(fitted_params_file='/Users/aparna.chaganty/adhoc-datascience/cmdstanpy/test/data/bernoulli_ppc/test_sampling_output-1.csv',
data='test/data/bernoulli.data.json', csv_basename='/Users/aparna.chaganty/adhoc-datascience/cmdstanpy/test/data/bernoulli_ppc/gen-quant-output')