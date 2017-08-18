import itertools
from random import random, sample

import networkx as nx

__all__ = ['min_fill_heuristic', 'min_width_heuristic', 'max_cardinality_heuristic',
           'is_simplicial', 'is_almost_simplicial',
           'treewidth_branch_and_bound', 'minor_min_width']


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
    Based on the algorithm presented in [GD]_

    """
    # we need only deal with the adjacency structure of G. We will also
    # be manipulating it directly so let's go ahead and make a new one
    adj = {v: set(G[v]) for v in G}

    lb = 0  # lower bound on treewidth
    while len(adj) > 1:

        # get the node with the smallest degree
        v = min(adj, key=lambda v: len(adj[v]))

        # find the vertex u such that the degree of u is minimal in the neighborhood of v
        neighbors = adj[v]

        def neighborhood_degree(u):
            Gu = adj[u]
            return sum(w in Gu for w in neighbors)
        u = min(neighbors, key=neighborhood_degree)

        # update the lower bound
        new_lb = len(adj[v])
        if new_lb > lb:
            lb = new_lb

        # contract the edge between u, v
        adj[v] = adj[v].union(n for n in adj[u] if n != v)
        for n in adj[v]:
            adj[n].add(v)
        for n in adj[u]:
            adj[n].discard(u)
        del adj[u]

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

    References
    ----------
    Based on the algorithm presented in [GD]_

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

    References
    ----------
    Based on the algorithm presented in [GD]_

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

    References
    ----------
    Based on the algorithm presented in [GD]_

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
    .. [GD] Gogate & Dechter, "A Complete Anytime Algorithm for Treewidth",
       https://arxiv.org/abs/1207.4109
    """
    # variable names are chosen to match the paper

    # our order will be stored in vector x, named to be consistent with
    # the paper
    x = []  # the partial order

    f = minor_min_width(G)  # our current lower bound guess, f(s) in the paper
    g = 0  # g(s) in the paper

    # we need the best current update we can find. best_found encodes the current
    # upper bound and the inducing order
    best_found = min_fill_heuristic(G)
    ub, __ = best_found

    # if our upper bound is the same as f, then we are done! Otherwise begin the
    # algorithm
    assert f <= ub, "Logic error"
    if f < ub:
        # we need only deal with the adjacency structure of G. We will also
        # be manipulating it directly so let's go ahead and make a new one
        adj = {v: set(G[v]) for v in G}

        best_found = _branch_and_bound(adj, x, g, f, best_found)

    return best_found


def _branch_and_bound(adj, x, g, f, best_found, skipable=set(), theorem6p1=None):

    if theorem6p1 is None:
        theorem6p1 = _theorem6p1()
    prune6p1, explored6p1 = theorem6p1

    # we'll need to know our current upper bound in several places
    ub, __ = best_found

    # ok, take care of the base case first
    if len(adj) < 2:
        # check if our current branch is better than the best we've already
        # found and if so update our best solution accordingly.
        if f < ub:
            return (f, x + list(adj))
        else:
            return best_found

    # so we have not yet reached the base case
    for n in adj:

        # we don't need to consider the neighbors of the last vertex eliminated
        if n in skipable:
            continue

        g_s = max(g, len(adj[n]))

        # according to Lemma 5.3, we can skip all of the neighbors of the last
        # variable eliniated when choosing the next variable
        next_skipable = adj[n]  # this does not get altered so we don't need a copy

        # update the state by eliminating n and adding it to the partial ordering
        adj_s = {v: adj[v].copy() for v in adj}  # create a new object
        _elim_adj(adj_s, n)
        x_s = x + [n]  # new partial ordering

        if prune6p1(x_s):
            continue

        # By Theorem 5.4, if any two vertices have ub + 1 common neighbors then
        # we can add an edge between them
        _theorem5p4(adj_s, ub)

        # ok, let's update our values
        f_s = max(g_s, minor_min_width(adj_s))

        g_s, f_s = _graph_reduction(adj_s, x_s, g_s, f_s)

        if f_s < ub:
            best_found = _branch_and_bound(adj_s, x_s, g_s, f_s, best_found,
                                           next_skipable, theorem6p1)
            ub, __ = best_found

    # let's store some information for pruning
    explored6p1(x)

    return best_found


def _graph_reduction(adj, x, g, f):
    """we can go ahead and remove any simplicial or almost-simplicial vertices from adj.
    """
    as_nodes = {v for v in adj if len(adj[v]) <= f and is_almost_simplicial(adj, v)}
    while as_nodes:
        for n in as_nodes:

            # update g and f
            dv = len(adj[n])
            if dv > g:
                g = dv
            if g > f:
                f = g

            # eliminate v
            x.append(n)
            _elim_adj(adj, n)

        # see if we have any more simplicial nodes
        as_nodes = {v for v in adj if len(adj[v]) <= f and is_almost_simplicial(adj, v)}

    return g, f


def _theorem5p4(adj, ub):
    """By Theorem 5.4, if any two vertices have ub + 1 common neighbors
    then we can add an edge between them.
    """
    new_edges = set()
    for u, v in itertools.combinations(adj, 2):
        if u in adj[v]:
            # already an edge
            continue

        if len(adj[u].intersection(adj[v])) > ub:
            new_edges.add((u, v))

    while new_edges:
        for u, v in new_edges:
            adj[u].add(v)
            adj[v].add(u)

        new_edges = set()
        for u, v in itertools.combinations(adj, 2):
            if u in adj[v]:
                continue

            if len(adj[u].intersection(adj[v])) > ub:
                new_edges.add((u, v))


def _theorem6p1():
    """See Theorem 6.1 in paper."""
    pruning_set = set()

    def _prune(x):
        if len(x) <= 2:
            return False
        key = (tuple(x[:-2]), x[-2], x[-1])  # this is faster than tuple(x[-3:])
        return key in pruning_set

    def _explored(x):
        if len(x) >= 3:
            prunable = (tuple(x[:-2]), x[-1], x[-2])
            pruning_set.add(prunable)

    return _prune, _explored
