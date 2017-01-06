import itertools
import networkx as nx

__all__ = ['treewidth_branch_and_bound', 'minor_min_width', 'min_width_heuristic', 'is_simplical',
           'is_complete', 'is_almost_simplical']


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

    if ub < h:
        raise Exception('logic error, upper bound should be greater than h')

    x = []  # the potential better ordering
    nv = []  # these are the neighbors of the last entry of x, empty right now
    g = 0
    return _BB(G, x, nv, ub, order, g, h, {})  # h == f in this case


def is_simplical(v, G):
    """Determines whether a vertex v in G is simplical.
    A vertex is simplical if its neighborhood induces a clique."""
    return is_complete(G.subgraph(G[v]))


def is_almost_simplical(v, G):
    """determines whether a vertex v in G is almost simplical.
    A vertex is almost simplical if all but one of its neighbors induce a clique"""
    for u in G[v]:
        if is_complete(G.subgraph([w for w in G[v] if w != u])):
            return True
    return False


def is_complete(G):
    """returns true if G is a complete graph"""
    n = len(G.nodes())  # get the number of nodes
    return len(G.edges()) == n*(n-1)/2


def _BB(G, x, nv, ub, order, g, f, graph_reductions):
    print 'G:', G.nodes()
    print 'x:', x
    print 'ub:', ub
    print 'f:', f
    if len(G.nodes()) < 2:
        return min(ub, f), x + G.nodes()  # ub, order

    for v in G:
        if len(x) > 0 and v in nv:  # we can skip these (section 5.2 in Gogate & Dechter)
            continue

        # create a new graph by making v simplical then removing it
        Gs = G.subgraph([n for n in G if n != v])
        for (n1, n2) in itertools.combinations(G[v], 2):
            Gs.add_edge(n1, n2)
        # Gs.remove_node(v)

        # append v to the order
        xs = x + [v]

        # get the values for our potential ordering
        gs = max(g, len(G[v]))
        hs = minor_min_width(Gs)
        fs = max(gs, hs)

        # we also note that for n1, n2 in G, if |intersection(N(n1), N(n2))| >= ub + 1
        # we should add an edge between n1 and n2. This will help us in the next step
        for n1, n2 in itertools.combinations(Gs.nodes(), 2):
            if (n1, n2) in Gs.edges() or (n2, n1) in Gs.edges():
                continue
            if len(set(Gs[n1]) & set(Gs[n2])) >= ub + 1:
                Gs.add_edge((n1, n2))

        # now we want to reduce the graph by eliminating all of the simplical or almost simplical
        # vertices in Gs
        # so we want v that are almost simplical and have degree less than lb (hs)
        # almost simplical means all but one of its neighbors induce a clique
        Gs, gs, fs, xs = _reduce_graph(Gs, gs, fs, hs, xs, graph_reductions)

        if fs < ub:
            # update with a better ordering
            ub, order = _BB(Gs, xs, G[v], ub, order, gs, fs, graph_reductions)

    # best order found so far
    return ub, order


def _reduce_graph(G, g, f, lb, x, graph_reductions):
    if len(G.nodes()) <= 2:
        G = nx.Graph()
        gs = max(g, 1)
        fs = max(gs, g)
        xs = x + G.nodes()
        return G, gs, fs, xs

    G_tuple = tuple(G.nodes())
    if G_tuple in graph_reductions:
        return graph_reductions[G_tuple]

    for v in G:
        if is_simplical(v, G):
            Gs = G.subgraph([n for n in G if n != v])

            gs = max(g, len(G[v]))
            fs = max(gs, f)
            xs = x + [v]
            graph_reductions[G_tuple] = _reduce_graph(Gs, gs, fs, lb, xs, graph_reductions)
            return graph_reductions[G_tuple]

        if is_almost_simplical(v, G) and (G.degree(v) <= lb):
            Gs = G.subgraph([n for n in G if n != v])
            for (n1, n2) in itertools.combinations(G[v], 2):
                Gs.add_edge(n1, n2)

            gs = max(g, len(G[v]))
            fs = max(gs, f)
            xs = x + [v]
            graph_reductions[G_tuple] = _reduce_graph(Gs, gs, fs, lb, xs, graph_reductions)
            return graph_reductions[G_tuple]

    graph_reductions[G_tuple] = (G, g, f, x)
    return graph_reductions[G_tuple]


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
