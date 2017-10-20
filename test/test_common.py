import os.path
import sys
import torch

_root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(-1, os.path.join(_root_dir, "pytorch", "test"))

from common import *

torch.set_default_tensor_type('torch.FloatTensor')
