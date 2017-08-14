import itertools
import random

import networkx as nx

__all__ = ['treewidth_branch_and_bound', 'minor_min_width', 'min_width_heuristic', 'is_simplicial',
           'is_almost_simplicial', 'min_fill_heuristic', 'max_cardinality_heuristic']


def treewidth_branch_and_bound(G, heuristic_function=min_width_heuristic):
    """Computes the treewidth of a graph G and a corresponding perfect elimination ordering.

    Parameters
    ----------
    G : graph
        A NetworkX graph.

    Returns
    -------
    treewidth : int
        The treewidth of the graph G.
    order : list
        An elimination order that induces the treewidth.

    References
    ----------
    .. [1] Gogate & Dechter, "A Complete Anytime Algorithm for Treewidth",
       https://arxiv.org/abs/1207.4109

    """
    # variable names were chosen to be consistent with the paper

    ub, order = heuristic_function(G)  # an upper bound on the treewidth
    lb = minor_min_width(G)  # a lower bound on the treewidth

    if lb == ub:
        return ub, order

    assert ub > lb, "Logic error"

    partial_order = []  # the potential better ordering
    nv = []  # these are the neighbors of v == partial_order[-1], empty for now

    upper_bound = (ub, order)
    state = (G.copy(), partial_order, nv)
    info = (lb, 0)  # lb, g is a variable in the paper
    return _BB(state, upper_bound, info, randomize)


def _BB(state, upper_bound, info, randomize=False):
    """helper function for treewidth_branch_and_bound
    NB: acts on G in place
    lb == f from paper
    """

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
        return min(ub, lb), partial_order + G.nodes()  # ub, order

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


def is_simplicial(G, n):
    """Determines whether a node n in G is simplicial.

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    n : node
        A node in G.

    Returns:
    is_simplicial : bool
        True if its neighbors form a clique.

    """
    return all(u in G[v] for u, v in itertools.combinations(G[n], 2))


def is_almost_simplicial(G, n):
    """Determines whether a node n in G is almost simplicial.

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    n : node
        A node in G.

    Returns:
    is_almost_simplicial : bool
        True if all but one of its neighbors induce a clique

    """
    for w in G[n]:
        if all(u in G[v] for u, v in itertools.combinations(G[n], 2) if u != w and v != w):
            return True
    return False


def is_complete(G):
    """Determines if G is a complete graph.

    Parameters
    ----------
    G : graph
        A NetworkX graph.

    Returns
    -------
    is_complete : bool
        True if G is a complete graph

    """
    n = len(G.nodes())  # get the number of nodes
    return len(G.edges()) == n * (n - 1) / 2


def minor_min_width(G):
    """Computes a lower bound on the treewidth of G.

    Parameters
    ----------
    G : graph
        A NetworkX graph.

    Returns
    -------
    lb : int
        A lower bound on the treewidth.

    References
    ----------
    .. [1] Gogate & Dechter, "A Complete Anytime Algorithm for Treewidth",
       https://arxiv.org/abs/1207.4109

    """
    G = G.copy()

    lb = 0  # lower bound on treewidth
    while len(G) > 1:

        # get the node with the smallest degree
        v = min(G, key=lambda v: len(G[v]))

        neighbors = G[v]

        # we can remove all of the singleton nodes without changing the lower bound
        if not neighbors:
            G.remove_node(v)
            continue

        # find the vertex u such that the degree of u is minimal in the neighborhood of v
        def neighborhood_degree(u):
            Gu = G[u]
            return sum(w in Gu for w in neighbors)
        u = min(neighbors, key=neighborhood_degree)

        # update the lower bound
        lb = max(lb, len(G[v]))

        # contract the edge between u, v
        G = nx.contracted_edge(G, (u, v), self_loops=False)

    return lb


def _elim(G, v):
    """Eliminates vertex v from graph G by making it simplicial then removing it.

    Notes:
        Acts on G in place.
    """
    G.add_edges_from(itertools.combinations(G[v], 2))
    G.remove_node(v)


def min_fill_heuristic(G, inplace=False):
    """Computes an upper bound on the treewidth of a graph based on the min-fill heuristic
    for the elimination ordering.

    Parameters
    ----------
    G : graph
        A NetworkX graph.

    inplace : bool
        If True, G will be made an empty graph in the process of
        running the function, otherwise the function uses a copy
        of G.

    Returns
    -------
    treewidth_upper_bound : int
        An upper bound on the treewidth of the graph G.

    order : list
        An elimination order that induces the treewidth.

    """
    if not inplace:
        G = G.copy()  # we will be destroying G

    num_nodes = len(G)

    order = [0] * num_nodes
    upper_bound = 0

    def _needed_edges(v):
        """The number of edges that would needed to be added to G to make v
        simplicial."""
        neighbors = G[v]
        n = len(neighbors)
        n_edges = n * (n - 1) // 2
        for u, w in itertools.combinations(neighbors, 2):
            if w in G[u]:
                n_edges -= 1
        return n_edges

    for i in range(num_nodes):
        # get the node that adds the fewest number of edges when eliminated from the graph
        # nodes are eliminated by making them simplicial, that is making their neighborhood
        # a clique
        v = min(G, key=_needed_edges)

        # if the number of neighbours of v is higher than upper_bound, update
        dv = len(G[v])
        if dv > upper_bound:
            upper_bound = dv

        # make v simplicial by making its neighborhood a clique then remove the node
        # add v to order
        _elim(G, v)
        order[i] = v

    return upper_bound, order


def min_width_heuristic(G, inplace=False):
    """Computes an upper bound on the treewidth of a graph based on
    the min-width heuristic for the elimination ordering.

    Parameters
    ----------
    G : graph
        A NetworkX graph.

    inplace : bool
        If True, G will be made an empty graph in the process of
        running the function, otherwise the function uses a copy
        of G.

    Returns
    -------
    treewidth_upper_bound : int
        An upper bound on the treewidth of the graph G.

    order : list
        An elimination order that induces the treewidth.

    """
    if not inplace:
        G = G.copy()  # we will be destroying G

    num_nodes = len(G)

    order = [0] * num_nodes
    upper_bound = 0

    for i in range(num_nodes):
        # get the node with the smallest degree
        v = min(G, key=lambda u: len(G[u]))

        # if the number of neighbours of v is higher than upper_bound, update
        dv = len(G[v])
        if dv > upper_bound:
            upper_bound = dv

        # make v simplicial by making its neighborhood a clique then remove the node
        # add v to order
        _elim(G, v)
        order[i] = v

    return upper_bound, order


def max_cardinality_heuristic(G, inplace=False):
    """Computes an upper bound on the treewidth of a graph based on
    the max-cardinality heuristic for the elimination ordering.

    Parameters
    ----------
    G : graph
        A NetworkX graph.

    inplace : bool
        If True, G will be made an empty graph in the process of
        running the function, otherwise the function uses a copy
        of G.

    Returns
    -------
    treewidth_upper_bound : int
        An upper bound on the treewidth of the graph G.

    order : list
        An elimination order that induces the treewidth.

    """
    if not inplace:
        G = G.copy()  # we will be destroying G

    num_nodes = len(G)

    if not num_nodes:
        return 0, []

    order = [0] * num_nodes
    upper_bound = 0

    v = random.choice(G.nodes())
    order[-1] = v
    labelled = {v}

    def n_labelled(v):
        # number of labelled neighbors
        if v in labelled:
            return -1
        n = 0
        for u in G[v]:
            if u in labelled:
                n += 1
        return n

    for i in range(num_nodes - 2, -1, -1):
        v = max(G, key=n_labelled)

        order[i] = v
        labelled.add(v)

    for v in order:
        # if the number of neighbours of v is higher than upper_bound, update
        dv = len(G[v])
        if dv > upper_bound:
            upper_bound = dv

        # make v simplicial by making its neighborhood a clique then remove the node
        # add v to order
        _elim(G, v)

    return upper_bound, order
