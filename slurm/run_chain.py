# User CmdStanPy to run one chain
# Required args:
# - cmdstanpath
# - model_exe
# - seed
# - chain_id
# - output_dir
# - data_file

import os
import sys

from cmdstanpy.model import CmdStanModel, set_cmdstan_path, cmdstan_path

useage = """\
run_chain.py <cmdstan_path> <model_exe> <seed> <chain_id> <output_dir> (<data_path>)\
"""

def main():
    if (len(sys.argv) < 5):
        print(missing arguments)
        print(useage)
        sys.exit(1)
    a_cmdstan_path = sys.argv[1]
    a_model_exe = sys.argv[2]
    a_seed = int(sys.argv[3])
    a_chain_id = int(sys.argv[4])
    a_output_dir = sys.argv[5]
    a_data_file = None
    if (len(sys.argv) > 6):
        a_data_file = sys.argv[6]

    set_cmdstan_path(a_cmdstan_path)
    print(cmdstan_path())

    mod = CmdStanModel(exe_file=a_model_exe)
    mod.sample(chains=1, chain_ids=a_chain_id, seed=a_seed, output_dir=a_output_dir, data_file=a_data_file)

if __name__ == "__main__":
    main()
