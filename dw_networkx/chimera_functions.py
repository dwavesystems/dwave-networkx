import networkx as nx


def get_chimera_dimensions(G):
    """determines whether a given graph is chimera structured"""

    if not nx.is_bipartite(G):
        raise ValueError('graph is not chimera-structured')

    L = max(G.degree(G.nodes()).values())

    print len(G)
    print [l_candidate for l_candidate in range(L, L-2, -1) if (len(G) % (2*l_candidate)) == 0]
