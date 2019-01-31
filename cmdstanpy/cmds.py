import json
import logging
import os
import os.path
import subprocess
import sys

from lib import Conf, Model
myconf = Conf()
cmdstan_path = myconf['cmdstan']

def _do_command(cmd, cwd=None):
    """Spawn process, get output/err/returncode.
    """
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        )
    proc.wait()
    stdout, stderr = proc.communicate()
    if stdout:
        print(stdout.decode('ascii').strip())
    if stderr:
        print('ERROR\n {} '.format(stderr.decode('ascii').strip()))
    if (proc.returncode):
        raise Exception('Command failed: {}'.format(cmd))

def preprocess_model(stan_file, hpp_file):
    """Invoke stanc on the given Stan model file.
    """
    stanc_path = os.path.join(cmdstan_path, 'bin', 'stanc')
    cmd = [stanc_path, '--o={}'.format(hpp_file), stan_file]
    _do_command(cmd)

def compile_model(stan_file, opt_lvl=3, overwrite=False):
    """Compile the given Stan model file to an executable.
    """
    if not os.path.exists(stan_file):
        raise Exception('no such stan_file {}'.format(stan_file))
        
    path = os.path.abspath(os.path.dirname(stan_file))
    program_name = os.path.basename(stan_file)
    model_name = program_name[:-5]
    hpp_name = model_name + ".hpp"
    hpp_file = os.path.join(path, hpp_name)
    if overwrite or not os.path.exists(hpp_file):
        try:
            preprocess_model(stan_file, hpp_file)
        except Exception:
            return Model(model_name, stan_file)

    target = os.path.join(path, model_name)
    if not overwrite and os.path.exists(target):
        print("model is up to date")
        return Model(model_name, stan_file, target)

    cmd = ['make', 'O={}'.format(opt_lvl), target]
    print(cmd)
    try:
        _do_command(cmd, cmdstan_path)
    except Exception:
        return Model(model_name, stan_file, target)
    return Model(model_name, stan_file)
