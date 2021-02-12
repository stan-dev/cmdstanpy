# distribute N chains, M chains per node on the cluster

# set up all slurm directives
# common args:
#  - cmdstan_path, model_exe, seed, output_dir, data_path
# unique arg:  chain_id  - jobs array number:  %a

#!/bin/bash
#SBATCH --job-name=cmdstanpy_runs
#SBATCH --output=cmdstanpy_stdout-%j-%a.out
#SBATCH --error=cmdstanpu_stderr-%j-%a.err
#SBATCH --nodes=20
#SBATCH --cpus-per-task=1
#SBATCH -a 0-100
python run_chain cmdstan_path model_exe seed chain_id output_dir data_path

