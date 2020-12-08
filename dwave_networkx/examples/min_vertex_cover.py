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

# The definition of a minimum vertex cover set is that each edge in the graph
# must have a vertex in the minimum vertex cover set, and we also want the
# vertex cover set to be as small as possible.

# Set up a Networkx Graph
G = nx.Graph()
G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (3, 5), (4, 5), (3, 6), (4, 7), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)])

# Get the minimum vertex cover, which is known in this case to be of
# length 5
candidate = dnx.min_vertex_cover(G, sampler)
if dnx.is_vertex_cover(G, candidate) and len(candidate) == 5:
    print (candidate, " is a minimum vertex cover")
else:
    print (candidate, " is not a minimum vertex cover")
