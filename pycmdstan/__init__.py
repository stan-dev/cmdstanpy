import logging
logging.basicConfig(level=logging.INFO)

# avoid _tkinter import error on readthedocs build
import os
if 'READTHEDOCS' in os.environ:
    import matplotlib
    matplotlib.use('agg')
    del matplotlib
del os

# reload submodules automatically
from importlib import reload
from . import io, model, viz, psis
[reload(_) for _ in (io, model, viz, psis)]
def reload():
    import importlib, pycmdstan
    importlib.reload(pycmdstan)

from .io import *
from .model import *
from .viz import *
from .psis import *
