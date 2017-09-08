import os.path
import sys

_root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(-1, os.path.join(_root_dir, "pytorch", "test"))

from common import *
