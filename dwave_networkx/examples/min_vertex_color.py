import dwave_networkx as dnx
import networkx as nx
import dimod

# Use basic simulated annealer
sampler = dimod.SimulatedAnnealingSampler()

# Set up a Networkx Graph
G = nx.Graph()
G.add_edges_from([(0,1), (0,2), (1,2), (1,3), (2,3), (1,4), (2,4), (3,4), (3,5), (4,5), (5,2)])

# Get the minimum vertex coloring, which is known in this case to be of
# length 6
candidate = dnx.min_vertex_coloring(G, sampler)
if dnx.is_vertex_coloring(G, candidate) and len(candidate) == 6:
  print str(candidate) + " is a minimum vertex coloring"
else: 
  print str(candidate) + " is not a minimum vertex coloring"
