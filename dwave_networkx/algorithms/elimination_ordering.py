import itertools
from random import random, sample

import networkx as nx

__all__ = ['min_fill_heuristic', 'min_width_heuristic', 'max_cardinality_heuristic',
           'is_simplicial', 'is_almost_simplicial',
           'treewidth_branch_and_bound', 'minor_min_width',
           'elimination_order_width']


def is_simplicial(G, n):
    """Determines whether a node n in G is simplicial.

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    n : node
        A node in G.

    Returns
    -------
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

    Returns
    -------
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

        if not neighbors:
            # if v is a singleton, then we can just delete it
            del adj[v]
            continue

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

    for i in range(num_nodes):
        # get the node that adds the fewest number of edges when eliminated from the graph
        v = min(adj, key=lambda x: _min_fill_needed_edges(adj, x))

        # if the number of neighbours of v is higher than upper_bound, update
        dv = len(adj[v])
        if dv > upper_bound:
            upper_bound = dv

        # make v simplicial by making its neighborhood a clique then remove the
        # node
        _elim_adj(adj, v)
        order[i] = v

    return upper_bound, order


def _min_fill_needed_edges(adj, n):
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
    """eliminates a variable, acting on the adj matrix of G,
    returning set of edges that were added.

    Parameters
    ----------
    adj: dict
        A dict of the form {v: neighbors, ...} where v are
        vertices in a graph and neighbors is a set.

    Returns
    ----------
    new_edges: set of edges that were added by eliminating v.

    """
    neighbors = adj[n]
    new_edges = set()
    for u, v in itertools.combinations(neighbors, 2):
        if v not in adj[u]:
            adj[u].add(v)
            adj[v].add(u)
            new_edges.add((u, v))
            new_edges.add((v, u))
    for v in neighbors:
        adj[v].discard(n)
    del adj[n]
    return new_edges


def elimination_order_width(G, order):
    """Calculates the width of the tree decomposition induced by a
    variable elimination order.

    Parameters
    ----------
    G : graph
        A NetworkX graph.

    order : list
        The elimination order. Must be a list of all of the variables
        in G.

    Returns
    -------
    treewidth : int
        The width of the tree decomposition induced by  order.

    """
    # we need only deal with the adjacency structure of G. We will also
    # be manipulating it directly so let's go ahead and make a new one
    adj = {v: set(G[v]) for v in G}

    treewidth = 0

    for v in order:

        # get the degree of the eliminated variable
        try:
            dv = len(adj[v])
        except KeyError:
            raise ValueError('{} is in order but not in G'.format(v))

        # the treewidth is the max of the current treewidth and the degree
        if dv > treewidth:
            treewidth = dv

        # eliminate v by making it simplicial (acts on adj in place)
        _elim_adj(adj, v)

    # if adj is not empty, then order did not include all of the nodes in G.
    if adj:
        raise ValueError('not all nodes in G were in order')

    return treewidth


def treewidth_branch_and_bound(G, elimination_order=None, treewidth_upperbound=None):
    """Computes the treewidth of a graph G and a corresponding perfect elimination ordering.

    Parameters
    ----------
    G : graph
        A NetworkX graph.

    elimination_order: list (optional)
        An elimination order. Uses the given elimination order as an
        initial best know order. If a good seed is provided, it may
        speed up computation. Default None, if not provided the initial
        order will be generated using the min fill heuristic.

    treewidth_upperbound : int (optional)
        Default None. An upper bound on the treewidth. Note that using
        this parameter can result in no returned order.

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
    # empty graphs have treewidth 0 and the nodes can be eliminated in
    # any order
    if not any(G[v] for v in G):
        return 0, list(G)

    # variable names are chosen to match the paper

    # our order will be stored in vector x, named to be consistent with
    # the paper
    x = []  # the partial order

    f = minor_min_width(G)  # our current lower bound guess, f(s) in the paper
    g = 0  # g(s) in the paper

    # we need the best current update we can find.
    ub, order = min_fill_heuristic(G)

    # if the user has provided an upperbound or an elimination order, check those against
    # our current best guess
    if elimination_order is not None:
        upperbound = elimination_order_width(G, elimination_order)
        if upperbound <= ub:
            ub, order = upperbound, elimination_order

    if treewidth_upperbound is not None and treewidth_upperbound < ub:
        # in this case the order might never be found
        ub, order = treewidth_upperbound, []

    # best found encodes the ub and the order
    best_found = ub, order

    # if our upper bound is the same as f, then we are done! Otherwise begin the
    # algorithm
    assert f <= ub, "Logic error"
    if f < ub:
        # we need only deal with the adjacency structure of G. We will also
        # be manipulating it directly so let's go ahead and make a new one
        adj = {v: set(G[v]) for v in G}

        best_found = _branch_and_bound(adj, x, g, f, best_found)

    return best_found


def _branch_and_bound(adj, x, g, f, best_found, skipable=set(), theorem6p2=None):
    """ Recursive branch and bound for computing treewidth of a subgraph.
    adj: adjacency list
    x: partial elimination order
    g: width of x so far
    f: lower bound on width of any elimination order starting with x
    best_found = ub,order: best upper bound on the treewidth found so far, and its elimination order
    skipable: vertices that can be skipped according to Lemma 5.3
    theorem6p2: terms that have been explored/can be pruned according to Theorem 6.2
    """

    # theorem6p2 checks for branches that can be pruned using Theorem 6.2
    if theorem6p2 is None:
        theorem6p2 = _theorem6p2()
    prune6p2, explored6p2, finished6p2 = theorem6p2
    # current6p2 is the list of prunable terms created during this instantiation of _branch_and_bound.
    # These terms will only be use during this call and its successors,
    # so they are removed before the function terminates.
    current6p2 = list()

    # theorem6p4 checks for branches that can be pruned using Theorem 6.4.
    # These terms do not need to be passed to successive calls to _branch_and_bound,
    # so they are simply created and deleted during this call.
    prune6p4, explored6p4 = _theorem6p4()

    # Note: theorem6p1 and theorem6p3 are a pruning strategies that are currently disabled
    # # as they does not appear to be invoked regularly,
    # and invoking it can require large memory allocations.
    # This can be fixed in the future if there is evidence that it's useful.
    # To add them in, define _branch_and_bound as follows:
    # def _branch_and_bound(adj, x, g, f, best_found, skipable=set(), theorem6p1=None,
    #                       theorem6p2=None, theorem6p3=None):

    # if theorem6p1 is None:
    #     theorem6p1 = _theorem6p1()
    # prune6p1, explored6p1 = theorem6p1

    # if theorem6p3 is None:
    #     theorem6p3 = _theorem6p3()
    # prune6p3, explored6p3 = theorem6p3

    # we'll need to know our current upper bound in several places
    ub, order = best_found

    # ok, take care of the base case first
    if len(adj) < 2:
        # check if our current branch is better than the best we've already
        # found and if so update our best solution accordingly.
        if f < ub:
            return (f, x + list(adj))
        elif f == ub and not order:
            return (f, x + list(adj))
        else:
            return best_found

    # so we have not yet reached the base case
    # Note: theorem 6.4 gives a heuristic for choosing order of n in adj.
    # Quick_bb suggests using a min-fill or random order.
    # We don't need to consider the neighbors of the last vertex eliminated
    sorted_adj = sorted((n for n in adj if n not in skipable), key=lambda x: _min_fill_needed_edges(adj, x))
    for n in sorted_adj:

        g_s = max(g, len(adj[n]))

        # according to Lemma 5.3, we can skip all of the neighbors of the last
        # variable eliniated when choosing the next variable
        next_skipable = adj[n]  # this does not get altered so we don't need a copy

        if prune6p2(x, n, next_skipable):
            continue

        # update the state by eliminating n and adding it to the partial ordering
        adj_s = {v: adj[v].copy() for v in adj}  # create a new object
        edges_n = _elim_adj(adj_s, n)
        x_s = x + [n]  # new partial ordering

        # pruning (disabled):
        # if prune6p1(x_s):
        #     continue

        if prune6p4(edges_n):
            continue

        # By Theorem 5.4, if any two vertices have ub + 1 common neighbors then
        # we can add an edge between them
        _theorem5p4(adj_s, ub)

        # ok, let's update our values
        f_s = max(g_s, minor_min_width(adj_s))

        g_s, f_s, as_list = _graph_reduction(adj_s, x_s, g_s, f_s)

        # pruning (disabled):
        # if prune6p3(x, as_list, n):
        #     continue

        if f_s < ub:
            best_found = _branch_and_bound(adj_s, x_s, g_s, f_s, best_found,
                                           next_skipable, theorem6p2=theorem6p2)
            # if theorem6p1, theorem6p3 are enabled, this should be called as:
            # best_found = _branch_and_bound(adj_s, x_s, g_s, f_s, best_found,
            #                                next_skipable, theorem6p1=theorem6p1,
            #                                theorem6p2=theorem6p2,theorem6p3=theorem6p3)
            ub, __ = best_found

        # store some information for pruning (disabled):
        # explored6p3(x, n, as_list)

        prunable = explored6p2(x, n, next_skipable)
        current6p2.append(prunable)

        explored6p4(edges_n)

    # store some information for pruning (disabled):
    # explored6p1(x)

    for prunable in current6p2:
        finished6p2(prunable)

    return best_found


def _graph_reduction(adj, x, g, f):
    """we can go ahead and remove any simplicial or almost-simplicial vertices from adj.
    """
    as_list = set()
    as_nodes = {v for v in adj if len(adj[v]) <= f and is_almost_simplicial(adj, v)}
    while as_nodes:
        as_list.union(as_nodes)
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

    return g, f, as_list


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


def _theorem6p2():
    """See Theorem 6.2 in paper.
    Prunes (x,...,a) when (x,a) is explored and a has the same neighbour set in both graphs.
    """
    pruning_set2 = set()

    def _prune2(x, a, nbrs_a):
        frozen_nbrs_a = frozenset(nbrs_a)
        for i in range(len(x)):
            key = (tuple(x[0:i]), a, frozen_nbrs_a)
            if key in pruning_set2:
                return True
        return False

    def _explored2(x, a, nbrs_a):
        prunable = (tuple(x), a, frozenset(nbrs_a))  # (s,a,N(a))
        pruning_set2.add(prunable)
        return prunable

    def _finished2(prunable):
        pruning_set2.remove(prunable)

    return _prune2, _explored2, _finished2


def _theorem6p3():
    """See Theorem 6.3 in paper.
    Prunes (s,b) when (s,a) is explored, b (almost) simplicial in (s,a), and a (almost) simplicial in (s,b)
    """
    pruning_set3 = set()

    def _prune3(x, as_list, b):
        for a in as_list:
            key = (tuple(x), a, b)  # (s,a,b) with (s,a) explored
            if key in pruning_set3:
                return True
        return False

    def _explored3(x, a, as_list):
        for b in as_list:
            prunable = (tuple(x), a, b)  # (s,a,b) with (s,a) explored
            pruning_set3.add(prunable)

    return _prune3, _explored3


def _theorem6p4():
    """See Theorem 6.4 in paper.
    Let E(x) denote the edges added when eliminating x. (edges_x below).
    Prunes (s,b) when (s,a) is explored and E(a) is a subset of E(b).
    For this theorem we only record E(a) rather than (s,E(a))
    because we only need to check for pruning in the same s context
    (i.e the same level of recursion).
    """
    pruning_set4 = list()

    def _prune4(edges_b):
        for edges_a in pruning_set4:
            if edges_a.issubset(edges_b):
                return True
        return False

    def _explored4(edges_a):
        pruning_set4.append(edges_a)  # (s,E_a) with (s,a) explored

    return _prune4, _explored4
