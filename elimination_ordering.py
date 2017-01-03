import itertools


def treewidth_branch_and_bound(G):
    """Computes the treewidth of a graph G and an elimination ordering"""
    pass


def min_width(G):
    """computes an upper bound on the treewidth of a graph based on the min-width heuristic
    for the elimination ordering.

    Gogate & Dechter, "A Complete Anytime Algorithm for Treewidth", https://arxiv.org/abs/1207.4109

    ub, order = min_width(G)
        G : a NetworkX graph
        ub : an upper bound on the treewidth of G
        order : an elimination ordering that induces that upper bound
    """
    G = G.copy()  # we will be manipulating G

    order = []
    upper_bound = 0

    while G.nodes():
        degreeG = G.degree(G.nodes())

        # get the node with the smallest degree
        degreeG = G.degree(G.nodes())
        v = min(degreeG, key=degreeG.get)

        # if the number of neighbours of v is higher then upper_bound, update
        upper_bound = max(upper_bound, len(G[v]))

        # make v simplical by adding edges between each of its neighbors
        for (n1, n2) in itertools.combinations(G[v], 2):
            G.add_edge(n1, n2)

        # remove the node from G and add it to the order
        order.append(v)
        G.remove_node(v)

    return upper_bound, order
