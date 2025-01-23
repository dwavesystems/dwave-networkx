********************
Elimination Ordering
********************

Many algorithms for NP-hard problems are exponential in treewidth. However,
finding a lower bound on treewidth is in itself NP-complete. [Gog2004]_
describes a branch-and-bound algorithm for computing the treewidth of an
undirected graph by searching in the space of *perfect elimination ordering* of
vertices of the graph.

A *clique* of a graph is a fully-connected subset of vertices; that is, every
pair of vertices in the clique share an edge. A *simplicial* vertex is one
whose neighborhood induces a clique. A perfect elimination ordering is an
ordering of vertices :math:`1..n` such that any vertex :math:`i` is simplicial
for the subset of vertices :math:`i..n`.

.. automodule:: dwave_networkx.algorithms.elimination_ordering
.. autosummary::
   :toctree: generated/

   chimera_elimination_order
   elimination_order_width
   is_almost_simplicial
   is_simplicial
   max_cardinality_heuristic
   minor_min_width
   min_fill_heuristic
   min_width_heuristic
   pegasus_elimination_order
   treewidth_branch_and_bound

