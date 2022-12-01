
def _add_compatible_edges(G, edge_list):
    # Check edge_list defines a subgraph of G and create subgraph.
    # Slow when edge_list is large, but clear (non-defaulted behaviour, so fine):
    if edge_list is not None:
        if not all(G.has_edge(*e) for e in edge_list):
            raise ValueError("edge_list contains edges incompatible with G")
        # Hard to check edge_list consistency owing to directedness, etc. Brute force
        G.remove_edges_from(list(G.edges))
        G.add_edges_from(edge_list)
        if G.number_of_edges() < len(edge_list):
            raise ValueError('edge_list contains duplicates.')

def _add_compatible_nodes(G, node_list):
    if node_list is not None:
        if not all(G.has_node(n) for n in node_list):
            raise ValueError("node_list contains nodes incompatible with G")
        nodes = set(node_list)
        remove_nodes = set(G) - nodes
        G.remove_nodes_from(remove_nodes)
        if G.number_of_nodes() < len(node_list):
            raise ValueError('node_list contains duplicates.')
        
def _add_compatible_terms(G, node_list, edge_list):
    _add_compatible_edges(G, edge_list)
    _add_compatible_nodes(G, node_list)
    #Check node deletion hasn't caused edge deletion:
    if edge_list is not None and len(edge_list) != G.number_of_edges():
        raise ValueError('The edge_list contains nodes absent from the node_list')
