import functools
import os
import unittest
import sys
import torch

_root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(-1, os.path.join(_root_dir, "pytorch", "test"))

from common import *

torch.set_default_tensor_type('torch.FloatTensor')


def skipIfTravis(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if os.getenv('TRAVIS'):
            raise unittest.SkipTest('Skip In Travis')
        return f(*args, **kwargs)
    return wrapper
