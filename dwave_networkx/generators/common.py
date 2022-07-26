
def _add_compatible_edges(G, edge_list):
    # Check edge_list defines a subgraph of G and create subgraph.
    # Slow when edge_list is large, but clear (non-defaulted behaviour, so fine):
    if edge_list is not None:
        if not all(G.has_edge(*e) for e in edge_list):
            raise ValueError("edge_list contains edges incompatible with a "
                             "fully yielded graph of the requested topology")
        # Hard to check edge_list consistency owing to directedness, etc. Brute force
        G.remove_edges_from(list(G.edges))
        G.add_edges_from(edge_list)
        if G.number_of_edges() < len(edge_list):
            raise ValueError('edge_list contains duplicates.')
