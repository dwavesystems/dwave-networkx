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

import dwave_networkx as dnx
import networkx as nx
import dimod

# Use basic simulated annealer
sampler = dimod.SimulatedAnnealingSampler()

# Set up a Networkx Graph
G = nx.Graph()
G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (1, 4), (2, 4), (3, 4), (3, 5), (4, 5), (5, 2)])

# Get the max cut
candidate = dnx.maximum_cut(G, sampler)
if len(candidate) == 3:
    print (candidate, " is the right length")
else:
    print (candidate, " is not the right length")
