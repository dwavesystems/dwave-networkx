import itertools
import random
import networkx as nx

__all__ = ['treewidth_branch_and_bound', 'minor_min_width', 'min_width_heuristic', 'is_simplicial',
           'is_complete', 'is_almost_simplicial', 'min_fill_heuristic']


def treewidth_branch_and_bound(G, randomize=False):
    """computes the treewidth of a graph G and a corresponding perfect elimination ordering.

    Gogate & Dechter, "A Complete Anytime Algorithm for Treewidth", https://arxiv.org/abs/1207.4109

    treewidth, order = treewidth_branch_and_bound(G)
        G : a NetworkX graph G
        treewidth : the treewidth of the graph G
        order : an elimination order that induces the treewidth
    """
    # variable names were chosen to be consistent with the paper

    ub, order = min_width_heuristic(G, randomize)  # an upper bound on the treewidth
    lb = minor_min_width(G)  # a lower bound on the treewidth

    if lb == ub:
        return ub, order

    if ub < lb:
        raise Exception('logic error, upper bound should be greater than h')

    partial_order = []  # the potential better ordering
    nv = []  # these are the neighbors of v == partial_order[-1], empty for now

    upper_bound = (ub, order)
    state = (G.copy(), partial_order, nv)
    info = (lb, 0)  # lb, g is a variable in the paper
    return _BB(state, upper_bound, info, randomize)


def _BB(state, upper_bound, info, randomize=False):
    """helper function for treewidth_branch_and_bound
    NB: acts on G in place
    lb == f from paper"""

    G, partial_order, nv = state  # extract the base graph and associated partial order
    ub, order = upper_bound
    lb, g = info

    # first up, we can add edges to G according to the following rule:
    # if |N(v1) and N(v2)| > ub then the final elimination order will require an edge
    # so let's be proactive and add them now
    for v1, v2 in itertools.combinations(G, 2):
        if len(tuple(nx.common_neighbors(G, v1, v2))) > ub:
            G.add_edge(v1, v2)

    # next we are going to remove some nodes from G and add them to the partial_order
    # specifically, we can remove simplicial or almost simplicial nodes from G and update
    # our lower bound
    sflag = True
    while sflag:
        sflag = False

        # we may want to sample in a random order
        if randomize:
            node_set = random.sample(G.nodes(), len(G))
        else:
            node_set = G.nodes()

        for v in sorted(node_set, key=G.degree):
            if is_simplicial(G, v) or (is_almost_simplicial(G, v) and G.degree(v) <= lb):
                sflag = True
                partial_order.append(v)
                g = max(g, G.degree(v))
                lb = max(g, lb)
                _elim(G, v)
                break

    # now the terminal rule
    if len(G.nodes()) < 2:
        return min(ub, lb), partial_order+G.nodes()  # ub, order

    # finally we try removing each of the variables from G and see which is the best
    for v in (randomize and random.sample(G.nodes(), len(G)) or G):
        if v in nv:  # we can skip direct neighbors
            continue

        # create a new state with v eliminated
        po_s = partial_order + [v]  # add v to the new partial order
        nv_s = G[v]  # the neighbors of v
        G_s = G.copy()  # so we can manipulate Gs
        _elim(G_s, v)

        new_state = (G_s, po_s, nv_s)

        g_s = max(g, G.degree(v))  # not changed by _elim
        lb_s = max(g_s, minor_min_width(G_s))
        new_info = (lb_s, g_s)

        if lb_s < ub:  # we need to keep going
            upper_bound = _BB(new_state, upper_bound, new_info, randomize)
            ub, order = upper_bound

    return upper_bound


def is_simplicial(G, v):
    """Determines whether a vertex v in G is simplicial.
    A vertex is simplicial if its neighbors form a clique."""
    return is_complete(G.subgraph(G[v]))


def is_almost_simplicial(G, v):
    """determines whether a vertex v in G is almost simplicial.
    A vertex is almost simplicial if all but one of its neighbors induce a clique"""
    for u in G[v]:
        if is_complete(G.subgraph([w for w in G[v] if w != u])):
            return True
    return False


def is_complete(G):
    """returns true if G is a complete graph"""
    n = len(G.nodes())  # get the number of nodes
    return len(G.edges()) == n*(n-1)/2


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

        if not Nv.nodes():  # in this case there is no edge to contract, happens in unconnected graphs
            G = G.copy()  # otherwise function would act on G in place
            G.remove_node(v)
            continue

        degreeNv = Nv.degree(Nv.nodes())
        u = min(degreeNv, key=degreeNv.get)

        # update the lower bound
        lb = max(lb, degreeG[v])

        # contract the edge between u, v
        G = nx.contracted_edge(G, (u, v), self_loops=False)

    return lb


def _elim(G, v):
    """eliminated vertex v from G by making it simplicial then removing it.
    NB: acts on G in place"""
    G.add_edges_from(itertools.combinations(G[v], 2))
    G.remove_node(v)


def min_fill_heuristic(G, randomize=False):
    """computes an upper bound on the treewidth of a graph based on the min-full heuristic
    for the elimination ordering

    ub, order = min_width(G)
        G : a NetworkX graph
        ub : an upper bound on the treewidth of G
        order : an elimination ordering that induces that upper bound
        randomize : variable order will not be fixed for G
    """

    # finds the number of edges that would need to be added in order to make the node's
    # neighborhood a clique
    needed_edges = lambda v: len(list(nx.non_edges(G.subgraph(G[v]))))

    G = G.copy()  # we will be manipulating G

    order = []  # variable elimination order
    upper_bound = 0

    while G.nodes():
        # get the node that adds the fewest number of edges when eliminated from the graph
        # nodes are eliminated by making them simplicial, that is making their neighborhood
        # a clique
        if randomize:
            # by default min takes the lowest index, this randomizes the order the values
            # are searched. The selected value will still have minimum degree
            v = min(random.sample(G.nodes(), len(G)), key=needed_edges)
        else:
            v = min(G.nodes(), key=needed_edges)

        # update the upper bound
        upper_bound = max(upper_bound, G.degree(v))

        # eliminate v by making it simplicial then removing it. Update the order
        order.append(v)
        _elim(G, v)

    return upper_bound, order


def min_width_heuristic(G, randomize=False):
    """computes an upper bound on the treewidth of a graph based on the min-width heuristic
    for the elimination ordering.

    ub, order = min_width_heuristic(G)
    ub, order = min_width_heuristic(G, randomize)
        G : a NetworkX graph
        ub : an upper bound on the treewidth of G
        order : an elimination ordering that induces that upper bound
        randomize : variable order will not be fixed for G
    """
    G = G.copy()  # we will be manipulating G in place

    order = []
    upper_bound = 0

    while G.nodes():
        # get the node with the smallest degree
        degreeG = G.degree(G.nodes())
        if randomize:
            # by default min takes the lowest index, this randomizes the order the values
            # are searched. The selected value will still have minimum degree
            v = min(random.sample(degreeG.keys(), len(degreeG)), key=degreeG.get)
        else:
            v = min(degreeG, key=degreeG.get)

        # if the number of neighbours of v is higher then upper_bound, update
        upper_bound = max(upper_bound, G.degree(v))

        # make v simplicial by making its neighborhood a clique then remove the node
        # add v to order
        _elim(G, v)
        order.append(v)

    return upper_bound, order
