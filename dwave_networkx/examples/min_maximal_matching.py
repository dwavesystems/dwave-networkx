from __future__ import print_function
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
