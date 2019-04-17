from utils import *
from lib import *
from cmds import *

examples_path = os.path.expanduser(os.path.join("~", "github", "stan-dev",
                                                    "cmdstanpy", "dev", "cmdstan", "examples", "bernoulli"))

myconf = Conf()
cmdstan_path = myconf['cmdstan']

bernstan = os.path.join(examples_path, "bernoulli.stan")
model = compile_model(bernstan)

jdata = os.path.join(examples_path, "bernoulli.data.json")
output = os.path.join(examples_path, "bernoulli.output")

args = SamplerArgs(model, seed=12345, post_warmup_draws=100, data_file=jdata, output_file=output, nuts_max_depth=11, adapt_delta=0.90)
transcript = os.path.join(examples_path, "bernoulli.samples")

runset = sample(model, chains=4, cores=2, seed=12345, post_warmup_draws_per_chain=100, data_file=jdata, csv_output_file=output, nuts_max_depth=11, adapt_delta=0.95)




