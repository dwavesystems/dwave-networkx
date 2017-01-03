import itertools
import networkx as nx


def treewidth_branch_and_bound(G):
    """computes the treewidth of a graph G and a corresponding perfect elimination ordering.

    Gogate & Dechter, "A Complete Anytime Algorithm for Treewidth", https://arxiv.org/abs/1207.4109

    treewidth, order = treewidth_branch_and_bound(G)
        G : a NetworkX graph G
        treewidth : the treewidth of the graph G
        order : an elimination order that induces the treewidth
    """
    ub, order = min_width_heuristic(G)  # an upper bound on the treewidth
    h = minor_min_width(G)  # a lower bound on the treewidth

    if h == ub:
        return ub, order

    raise NotImplementedError('remainder of the algorithm is not yet implemented')


def minor_min_width(G):
    """computes a lower bound on the treewidth of G.

    Gogate & Dechter, "A Complete Anytime Algorithm for Treewidth", https://arxiv.org/abs/1207.4109

    lb = minor_min_width(G)
        G : a NetworkX graph
        lb : a lower bound on the treewidth
    """
    lb = 0
    while len(G.nodes()) > 1:
        # get the node with the smallest degree
        degreeG = G.degree(G.nodes())
        v = min(degreeG, key=degreeG.get)

        # find the vertex u such that the degree of u is minimal in the neighborhood of v
        Nv = G.subgraph(G[v].keys())

        degreeNv = Nv.degree(Nv.nodes())
        u = min(degreeNv, key=degreeNv.get)

        # update the lower bound
        lb = max(lb, degreeG[v])

        # contract the edge between u, v
        G = nx.contracted_edge(G, (u, v), self_loops=False)

    return lb


def min_width_heuristic(G):
    """computes an upper bound on the treewidth of a graph based on the min-width heuristic
    for the elimination ordering.

    ub, order = min_width(G)
        G : a NetworkX graph
        ub : an upper bound on the treewidth of G
        order : an elimination ordering that induces that upper bound
    """
    G = G.copy()  # we will be manipulating G

    order = []
    upper_bound = 0

    while G.nodes():
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
