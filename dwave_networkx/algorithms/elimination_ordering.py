import itertools
import random
from random import random

import networkx as nx

__all__ = ['treewidth_branch_and_bound2', 'minor_min_width', 'min_width_heuristic', 'is_simplicial',
           'is_almost_simplicial', 'min_fill_heuristic', 'max_cardinality_heuristic',
           'treewidth_branch_and_bound']


def treewidth_branch_and_bound(G):
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

    ub, order = min_width_heuristic(G)  # an upper bound on the treewidth
    lb = minor_min_width(G)  # a lower bound on the treewidth

    if lb == ub:
        return ub, order

    assert ub > lb, "Logic error"

    partial_order = []  # the potential better ordering
    nv = []  # these are the neighbors of v == partial_order[-1], empty for now

    upper_bound = (ub, order)
    state = (G.copy(), partial_order, nv)
    info = (lb, 0)  # lb, g is a variable in the paper
    return _BB(state, upper_bound, info)


def _BB(state, upper_bound, info):
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
    for v in G:
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
            upper_bound = _BB(new_state, upper_bound, new_info)
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


def min_fill_heuristic(G):
    """Computes an upper bound on the treewidth of a graph based on
    the min-fill heuristic for the elimination ordering.

    Parameters
    ----------
    G : graph
        A NetworkX graph.

    Returns
    -------
    treewidth_upper_bound : int
        An upper bound on the treewidth of the graph G.

    order : list
        An elimination order that induces the treewidth.

    """
    # we need only deal with the adjacency structure of G. We will also
    # be manipulating it directly so let's go ahead and make a new one
    adj = {v: set(G[v]) for v in G}

    num_nodes = len(adj)

    # preallocate the return values
    order = [0] * num_nodes
    upper_bound = 0

    def _needed_edges(n):
        # determines how many edges would needed to be added to G in order
        # to make node n simplicial.
        e = 0  # number of edges needed
        for u, v in itertools.combinations(adj[n], 2):
            if u not in adj[v]:
                e += 1
        # We add random() which picks a value in the range [0., 1.). This is ok because the
        # e are all integers. By adding a small random value, we randomize which node is
        # chosen without affecting correctness.
        return e + random()

    for i in range(num_nodes):
        # get the node that adds the fewest number of edges when eliminated from the graph
        v = min(adj, key=_needed_edges)

        # if the number of neighbours of v is higher than upper_bound, update
        dv = len(adj[v])
        if dv > upper_bound:
            upper_bound = dv

        # make v simplicial by making its neighborhood a clique then remove the
        # node
        _elim_adj(adj, v)
        order[i] = v

    return upper_bound, order


def min_width_heuristic(G):
    """Computes an upper bound on the treewidth of a graph based on
    the min-width heuristic for the elimination ordering.

    Parameters
    ----------
    G : graph
        A NetworkX graph.

    Returns
    -------
    treewidth_upper_bound : int
        An upper bound on the treewidth of the graph G.

    order : list
        An elimination order that induces the treewidth.

    """
    # we need only deal with the adjacency structure of G. We will also
    # be manipulating it directly so let's go ahead and make a new one
    adj = {v: set(G[v]) for v in G}

    num_nodes = len(adj)

    # preallocate the return values
    order = [0] * num_nodes
    upper_bound = 0

    for i in range(num_nodes):
        # get the node with the smallest degree. We add random() which picks a value
        # in the range [0., 1.). This is ok because the lens are all integers. By
        # adding a small random value, we randomize which node is chosen without affecting
        # correctness.
        v = min(adj, key=lambda u: len(adj[u]) + random())

        # if the number of neighbours of v is higher than upper_bound, update
        dv = len(adj[v])
        if dv > upper_bound:
            upper_bound = dv

        # make v simplicial by making its neighborhood a clique then remove the
        # node
        _elim_adj(adj, v)
        order[i] = v

    return upper_bound, order


def max_cardinality_heuristic(G):
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
    # we need only deal with the adjacency structure of G. We will also
    # be manipulating it directly so let's go ahead and make a new one
    adj = {v: set(G[v]) for v in G}

    num_nodes = len(adj)

    # preallocate the return values
    order = [0] * num_nodes
    upper_bound = 0

    # we will need to track the nodes and how many labelled neighbors
    # each node has
    labelled_neighbors = {v: 0 for v in adj}

    # working backwards
    for i in range(num_nodes):
        # pick the node with the most labelled neighbors
        v = max(labelled_neighbors, key=lambda u: labelled_neighbors[u] + random())
        del labelled_neighbors[v]

        # increment all of its neighbors
        for u in adj[v]:
            if u in labelled_neighbors:
                labelled_neighbors[u] += 1

        order[-(i + 1)] = v

    for v in order:
        # if the number of neighbours of v is higher than upper_bound, update
        dv = len(adj[v])
        if dv > upper_bound:
            upper_bound = dv

        # make v simplicial by making its neighborhood a clique then remove the node
        # add v to order
        _elim_adj(adj, v)

    return upper_bound, order


def _elim(G, v):
    """Eliminates vertex v from graph G by making it simplicial then removing it.

    Notes:
        Acts on G in place.
    """
    G.add_edges_from(itertools.combinations(G[v], 2))
    G.remove_node(v)


def _elim_adj(adj, n):
    """eliminates a variable, acting on the adj matrix of G.

    Parameters
    ----------
        adj: dict
            A dict of the form {v: neighbors, ...} where v are
            vertices in a graph and neighbors is a set.
    """
    neighbors = adj[n]
    for u, v in itertools.combinations(neighbors, 2):
        adj[u].add(v)
        adj[v].add(u)
    for v in neighbors:
        adj[v].discard(n)
    del adj[n]


def treewidth_branch_and_bound2(G, heuristic_function=min_fill_heuristic):
    # TODO

    G = G.copy()
    partial_order = []

    lb = minor_min_width(G)
    g = 0  # g(s) in the paper

    ub, order = heuristic_function(G)

    if ub == lb:
        return ub, order

    assert lb < ub, "Logic Error"

    best_found = (ub, order)

    return _branch_and_bound(G, partial_order, best_found, g, lb)


def _branch_and_bound(G, x, best_found, g, f,
                      last_vertex_neighbors=[]):

    # ok, take care of the base case first
    ub, __ = best_found
    if len(G) < 2:

        # check if our current branch is better than the best we've already
        # found and if so update our best solution accordingly.
        if f < ub:
            x = x + G.nodes()  # create a new object
            best_found = (f, x)

        # return in either case
        return best_found

    for v in G:

        # we don't need to consider the neighbors of the last vertex eliminated
        if v in last_vertex_neighbors:
            continue

        gs = max(g, len(G[v]))

        Gs = G.copy()
        vertex_neighbors = Gs[v]
        _elim(Gs, v)
        xs = x + [v]

        # at this point, if any two nodes u,w in Gs have more than ub
        # common neighbors, then we can connect them with an edge
        edges = [(u, w) for u, w in itertools.combinations(Gs, 2)
                 if u not in Gs[w] and len(set(Gs[u]).intersection(Gs[w])) > ub]
        while edges:
            for u, w in edges:
                Gs.add_edge(u, w)
            edges = [(u, w) for u, w in itertools.combinations(Gs, 2)
                     if u not in Gs[w] and len(set(Gs[u]).intersection(Gs[w])) > ub]

        # ok, let's update our values
        fs = max(gs, minor_min_width(Gs))

        # we can go ahead and remove any simplicial or almost-simplicial vertices from Gs
        almost_simplicial = [u for u in Gs if len(Gs[u]) < fs and is_almost_simplicial(Gs, u)]
        while almost_simplicial:
            for u in almost_simplicial:
                gs = max(gs, len(Gs[u]))
                fs = max(gs, fs)
                # we don't need to copy Gs again here
                _elim(Gs, u)
                x.append(u)

            almost_simplicial = [u for u in Gs if len(Gs[u]) <= fs and is_almost_simplicial(Gs, u)]

        ub, __ = best_found
        if fs < ub:
            # print fs, ub
            best_found = _branch_and_bound(Gs, xs, best_found, g, fs, vertex_neighbors)

    return best_found
