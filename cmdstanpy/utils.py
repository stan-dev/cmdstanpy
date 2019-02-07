"""
Utility functions
"""

import os
import os.path
import subprocess
import numpy as np

def is_int(i):
    try:
       test = int(i)
    except Exception:
        return False
    return True

def do_sample(runset, idx):
    """Spawn process, capture stdout and std err to transcript file, return returncode.
    """
    cmd = runset.cmds[idx]
    proc = subprocess.Popen(
        cmd.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        )
    proc.wait()
    stdout, stderr = proc.communicate()
    transcript_file = runset.transcript_files[idx]
    with open(transcript_file, "w+") as transcript:
        if stdout:
            transcript.write(stdout.decode('ascii'))
        if stderr:
            transcript.write('ERROR')
            transcript.write(stderr.decode('ascii'))
    runset.set_retcode(idx, proc.returncode)


def do_command(cmd, cwd=None):
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

def _rdump_array(key, val):
    c = 'c(' + ', '.join(map(str, val.T.flat)) + ')'
    if (val.size, ) == val.shape:
        return '{key} <- {c}'.format(key=key, c=c)
    else:
        dim = '.Dim = c{0}'.format(val.shape)
        struct = '{key} <- structure({c}, {dim})'.format(key=key, c=c, dim=dim)
        return struct


def rdump(path, data):
    """Dump a dict of data to a R dump format file.
    """
    with open(path, 'w') as fd:
        for key, val in data.items():
            if isinstance(val, np.ndarray) and val.size > 1:
                line = _rdump_array(key, val)
            elif isinstance(val, list) and len(val) > 1:
                line = _rdump_array(key, np.asarray(val))
            else:

                try:
                    val = val.flat[0]
                except AttributeError:
                    pass
                line = '%s <- %s' % (key, val)
            fd.write(line)
            fd.write('\n')
