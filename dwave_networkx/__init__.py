from __future__ import absolute_import

import sys
_PY2 = sys.version_info[0] == 2

import dwave_networkx.architectures
from dwave_networkx.architectures import *

try:
    import dwave_networkx.x_architectures
    from dwave_networkx.x_architectures import *
except ImportError:
    pass

import dwave_networkx.algorithms
from dwave_networkx.algorithms import *

import dwave_networkx.utils
from dwave_networkx.exceptions import *


import dwave_networkx.default_sampler
from dwave_networkx.default_sampler import *

import dwave_networkx.drawing
from dwave_networkx.drawing import *

__version__ = '0.5.0'
__author__ = 'D-Wave Systems Inc.'
__authoremail__ = 'acondello@dwavesys.com'
__description__ = 'A NetworkX extension providing graphs and algorithms relevent to working with the D-Wave System'
