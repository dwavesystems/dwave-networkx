# Copyright 2019 D-Wave Systems Inc.
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

import itertools
import collections.abc as abc

from collections import namedtuple

import networkx as nx


__all__ = 'markov_network',


###############################################################################
# The following code is partially based on https://github.com/tbabej/gibbs
#
# MIT License
# ===========
#
# Copyright 2017 Tomas Babej
# https://github.com/tbabej/gibbs
#
# This software is released under MIT licence.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#


def markov_network(potentials):
    """Creates a Markov Network from potentials.

    A Markov Network is also knows as a `Markov Random Field`_

    Parameters
    ----------
    potentials : dict[tuple, dict]
        A dict where the keys are either nodes or edges and the values are a
        dictionary of potentials. The potential dict should map each possible
        assignment of the nodes/edges to their energy.

    Returns
    -------
    MN : :obj:`networkx.Graph`
        A markov network as a graph where each node/edge stores its potential
        dict as above.

    Examples
    --------
    >>> potentials = {('a', 'b'): {(0, 0): -1,
    ...                            (0, 1): .5,
    ...                            (1, 0): .5,
    ...                            (1, 1): 2}}
    >>> MN = dnx.markov_network(potentials)
    >>> MN['a']['b']['potential'][(0, 0)]
    -1

    .. _Markov Random Field: https://en.wikipedia.org/wiki/Markov_random_field

    """

    G = nx.Graph()

    G.name = 'markov_network({!r})'.format(potentials)

    # we use 'clique' because the keys of potentials can be either nodes or
    # edges, but in either case they are fully connected.
    for clique, phis in potentials.items():

        num_vars = len(clique)

        # because this data potentially wont be used for a while, let's do some
        # input checking now and save some debugging issues later
        if not isinstance(phis, abc.Mapping):
            raise TypeError("phis should be a dict")
        elif not all(config in phis for config in itertools.product((0, 1), repeat=num_vars)):
            raise ValueError("not all potentials provided for {!r}".format(clique))

        if num_vars == 1:
            u, = clique
            G.add_node(u, potential=phis)
        elif num_vars == 2:
            u, v = clique
            # in python<=3.5 the edge order might not be consistent so we store
            # the relevant order of the variables relative to the potentials
            G.add_edge(u, v, potential=phis, order=(u, v))
        else:
            # developer note: in principle supporting larger cliques can be done
            # using higher-order, but it would make the use of networkx graphs
            # far more difficult
            raise ValueError("Only supports cliques up to size 2")

    return G
