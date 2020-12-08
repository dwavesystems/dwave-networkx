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

import networkx as nx 
import dwave_networkx as dnx 
import dimod

# Use basic simulated annealer
sampler = dimod.SimulatedAnnealingSampler()

# Set up a Networkx Graph
G = nx.Graph()
G.add_edges_from([(1,2),(1,3),(2,3),(3,4),(3,5),(4,5),(4,6),(5,6),(6,7)])

# Find the maximum independent set, which is known in this case to be of length 3
candidate = dnx.maximum_independent_set(G, sampler)
if dnx.is_independent_set(G, candidate) and len(candidate) == 3:
    print(candidate, " is a maximum independent set")
else:
    print(candidate, " is not a minimum vertex coloring")