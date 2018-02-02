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
  print str(candidate) + " is a minimum vertex cover"
else: 
  print str(candidate) + " is not a minimum vertex cover"
