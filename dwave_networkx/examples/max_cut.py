import dwave_networkx as dnx
import networkx as nx
import dimod

# Use basic simulated annealer
sampler = dimod.SimulatedAnnealingSampler()

# Set up a Networkx Graph
G = nx.Graph()
G.add_edges_from([(0,1), (0,2), (1,2), (1,3), (2,3), (1,4), (2,4), (3,4), (3,5), (4,5), (5,2)])

# Get the max cut
candidate = dnx.maximum_cut(G, sampler)
if len(candidate) == 3:
  print str(candidate) + " is the right length"
else: 
  print str(candidate) + " is not the right length"
