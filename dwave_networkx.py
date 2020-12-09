# Copyright 2020 D-Wave Systems Inc.
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

from dwave.plugins.networkx import *
from dwave.plugins.networkx import (
    __version__,
    __author__,
    __authoremail__,
    __description__,
    )


# We want `dwave_networkx` to alias `dwave.plugins.networkx` so we mess around
# with the path. We use a function to make cleaning up this namespace easier.
def alias_subpackages():
    import importlib
    import pkgutil
    import sys

    import warnings
    warnings.warn("the dwave_networkx namespace was deprecated in "
                  "dwave-networkx 0.9.0, please use "
                  "dwave.plugins.networkx instead.",
                  DeprecationWarning, stacklevel=3)

    for module in pkgutil.walk_packages(dwave.plugins.networkx.__path__,
                                        dwave.plugins.networkx.__name__ + '.'):
        # only want the subpackages
        if not module.ispkg:
            continue

        # pretend that each subpackage lives in the dwave_networkx namespace
        package_name = 'dwave_networkx.' + module.name.split('.')[-1]
        sys.modules[package_name] = importlib.import_module(module.name)


alias_subpackages()
del alias_subpackages
