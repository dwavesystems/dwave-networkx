def graphs_equal(G, H):
    return type(G) == type(H) and {*G} == {*H} and len(G.edges) == len(H.edges) and all(H.has_edge(*e) for e in G.edges)
