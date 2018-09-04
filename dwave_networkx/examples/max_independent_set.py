from __future__ import print_function
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