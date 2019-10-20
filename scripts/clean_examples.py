import glob
import os
from cmdstanpy import cmdstan_path

def clean_examples():
    cmdstan_examples = s.path.join(cmdstan_path(), "examples")
    for root, _, files in os.walk(cmdstan_examples)
        for filename in files:
            _, ext = os.path.splitext(filename)
            if ext.lower() in (".o", ".hpp", ".exe", ""):
                os.remove(os.path.join(root, filename))

if __name__ == "__main__":
    cmdstan_examples()
