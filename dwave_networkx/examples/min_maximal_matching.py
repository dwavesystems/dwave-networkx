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
import dimod

# Use basic simulated annealer
sampler = dimod.SimulatedAnnealingSampler()

G = dnx.chimera_graph(1, 1, 4)
# Get the minimum maximal matching, which is known in this case to be of
# length 4
candidate = dnx.min_maximal_matching(G, sampler)

if dnx.is_maximal_matching(G, candidate) and len(candidate) == 4:
    print (candidate, " is a minimum maximal matching")
else:
    print (candidate, " is not a minimum maximal matching")
