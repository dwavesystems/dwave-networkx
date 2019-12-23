============
Introduction
============

For quantum computing, as for classical, the first step in
solving a problem is to express it in a mathematical formulation
compatible with the underlying physical hardware.

Native Formulations for D-Wave Systems
======================================

D-Wave systems solve problems that can be mapped onto an Ising model or a
quadratic unconstrained binary optimization (QUBO) problem.

.. math::

  \text{Ising:} \qquad
  E(\pmb{s}|\pmb{h},\pmb{J})
  = \left\{ \sum_{i=1}^N h_i s_i +
  \sum_{i<j}^N J_{i,j} s_i s_j  \right\}
  \qquad\qquad s_i\in\{-1,+1\}

is an objective function of :math:`N` variables :math:`\pmb s=[s_1,...,s_N]`
corresponding to physical Ising spins, where :math:`h_i` are the biases and
:math:`J_{i,j}` the couplings between spins.

.. math::

		\text{QUBO:} \qquad E(\pmb{x}| \pmb{Q})
    =  \sum_{i\le j}^N x_i Q_{i,j} x_j
    \qquad\qquad x_i\in \{0,1\}

is an objective function of :math:`N` binary variables represented as an
upper-diagonal matrix :math:`Q`, where diagonal terms are the linear coefficients and
the nonzero off-diagonal terms the quadratic coefficients.

Objective functions can be represented by graphs, a collection
of nodes (representing variables) and the connections between them (edges).

D-Wave QPU Topology
===================

To solve a QUBO or Ising objective function on the D-Wave system, you
must map it to a graph that represents the topology of the system's
qubits. For D-Wave 2000Q and 2X systems, this is the *chimera* topology; for next-generation
systems, this is the *Pegasus* topology.

Chimera
-------

The Chimera architecture comprises sets of connected unit cells, each with four
horizontal qubits connected to four vertical qubits via couplers (bipartite
connectivity). Unit cells are tiled vertically and horizontally with adjacent
qubits connected, creating a lattice of sparsely connected qubits. A unit cell
is typically rendered as either a cross or a column.

.. figure:: _images/ChimeraUnitCell.png
	:align: center
	:name: ChimeraUnitCell
	:scale: 40 %
	:alt: Chimera unit cell.

	Chimera unit cell.


.. figure:: _images/chimera.png
  :name: chimera
  :scale: 70 %
  :alt: Chimera graph.  qubits are arranged in unit cells that form bipartite connections.

  A :math:`3 {\rm x} 3`  Chimera graph, denoted C3. Qubits are arranged in 9 unit cells.

Chimera qubits are considered to have a nominal length of 4 (each qubit
is connected to 4 orthogonal qubits through internal couplers) and degree of 6 (each qubit
is coupled to 6 different qubits).

The notation CN refers to a Chimera graph consisting of an :math:`N{\rm x}N` grid of unit cells.
The D-Wave 2000Q QPU supports a C16 Chimera graph: its 2048 qubits are logically mapped into a
:math:`16 {\rm x} 16` matrix of unit cells of 8 qubits.

Pegasus
-------

In Pegasus as in Chimera, qubits are “oriented” vertically or horizontally but similarly aligned
qubits can also be also shifted by distances and in groupings that differ between Pegasus families.
Pegasus qubits are also more densely connected and have three types of coupler:

- *Internal couplers*.
  Internal couplers connect pairs of orthogonal (with opposite orientation) qubits. In Pegasus,
  each qubit is connected via internal coupling to 12 other qubits (versus four in the Chimera topology).
- *External couplers*.
  External couplers connect vertical qubits to adjacent vertical qubits and horizontal
  qubits to adjacent horizontal qubits. Each qubit has one or two external couplers.
- *Odd couplers*.
  Odd couplers connect similarly aligned pairs of qubits. Each qubit has one odd coupler.

.. figure:: _images/pegasus_qubits.png
	:align: center
	:name: pegasus_qubits
	:scale: 100 %
	:alt: Pegasus qubits

	Pegasus qubits. Qubits are drawn as horizontal and vertical loops. The horizontal qubit in the center, shown with its odd coupler in red and numbered 1, is internally coupled to vertical qubits, in pairs 3 through 8, each pair and its odd coupler shown in a different color, and externally coupled to horizontal qubits 2 and 9, each shown in a different color.

.. figure:: _images/pegasus_roadway.png
	:align: center
	:name: pegasus_roadway
	:scale: 100 %
	:alt: Pegasus roadway graphic

	Pegasus qubits. Qubits in this "roadway" graphic are represented as dots and couplers as lines. The top qubit in the center, shown in red and numbered 1, is oddly coupled to the (red) qubit shown directly below it, internally coupled to vertical qubits, in pairs 3 through 8, each pair and its odd coupler shown in a different color, and externally coupled to horizontal qubits 2 and 9, each shown in a different color.

Pegasus qubits are considered to have a nominal length of 12 (each qubit is connected to
12 orthogonal qubits through internal couplers) and degree of 15 (each qubit is coupled to
15 different qubits).

As we use the notation CN to refer to a Chimera graph with size parameter N, we refer to instances
of Pegasus topologies by PN; for example, P3 is a graph with 144 nodes.

D-Wave NetworkX
===============

D-Wave NetworkX provides tools for working with Chimera and Pegasus graphs and
implementations of graph-theory algorithms on the D-Wave system and other binary
quadratic model samplers; for example, functions such as `draw_chimera()` provide
easy visualization for Chimera graphs; functions such as `maximum_cut()` or
`min_vertex_cover()` provide graph algorithms useful to optimization problems
that fit well with the D-Wave system.

Like the D-Wave system, all other supported samplers (a process that samples
from low energy states of the problem's objective function) must have
`sample_qubo` and `sample_ising` methods for solving Ising and QUBO models
and return an iterable of samples in order of increasing energy. You can set
a default sampler using the `set_default_sampler()` function.

Below you can see how to create Chimera graphs implemented in the D-Wave 2X and D-Wave 2000Q systems:

.. code:: python

  import dwave_networkx as dnx

  # D-Wave 2X
  C = dnx.chimera_graph(12, 12, 4)

  # D-Wave 2000Q
  C = dnx.chimera_graph(16, 16, 4)
