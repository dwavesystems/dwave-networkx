# Copyright 2018 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from warnings import warn as _warn

import dwave_networkx.generators
from dwave_networkx.generators import *

import dwave_networkx.algorithms
from dwave_networkx.algorithms import *

import dwave_networkx.utils
from dwave_networkx.exceptions import *

import dwave_networkx.default_sampler
from dwave_networkx.default_sampler import *

import dwave_networkx.drawing
from dwave_networkx.drawing import *

from dwave_networkx.package_info import __version__, __author__, \
    __authoremail__, __description__


_warn("dwave-networkx is deprecated and will be replaced by dwave-graphs in Ocean 10. "
      "Most functionality previously provided by dwave-networkx is now available "
      "as part of dwave-graphs under the 'dwave.graphs' namespace.",
      category=DeprecationWarning,
      stacklevel=2)
