import matplotlib.pyplot as plt
import dwave_networkx as dnx
import networkx as nx

G = dnx.chimera_graph(2, 2, 4)
dnx.draw_chimera(G)
plt.show()
