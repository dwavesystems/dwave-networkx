# Copyright 2026 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import itertools
import unittest

import networkx as nx
import numpy as np

from dwave_networkx import zephyr_graph
from dwave_networkx.generators._zephyr_playground import (
    ZephyrSearchMetadata, zephyr_quotient_search)


def generate_faulty_zephyr_graph(m, t, proportion, uniform_proportion, seed=None):
    """Create a Zephyr graph with simulated hardware faults.

    Nodes are deleted in two phases: (1) ``round(proportion * uniform_proportion * N)`` nodes are
    chosen uniformly at random and removed; (2) ``round(proportion * (1 - uniform_proportion) * N)``
    additional nodes are removed iteratively, one node at a time.

    During phase (2), for each candidate node ``v`` we compute
    ``r(v) = sum(dist(v, d) for d in D)``, where ``D`` is the current set of deleted nodes and
    ``dist`` is shortest-path distance in the original (unfaulted) graph. The next deleted node is
    sampled with probability proportional to ``1 / r(v)``. After each deletion, distances are
    updated by adding shortest-path contributions from the newly deleted node, so probabilities are
    re-evaluated at every iteration. This makes nodes near multiple already deleted nodes more
    likely to fail than nodes near fewer deleted nodes.

    Nodes that are unreachable from at least one deleted node have zero weight and are not selected.
    The two phases remove approximately ``proportion`` of all nodes.

    Args:
        m (int): Zephyr row count.
        t (int): Zephyr tile count.
        proportion (float): Total fraction of nodes to remove, in ``(0, 1)``.
        uniform_proportion (float): Fraction of removed nodes that are chosen
            uniformly (the complementary fraction is chosen by distance-based
            sampling).
        seed (int | None): RNG seed for reproducibility. Defaults to ``None``.

    Returns:
        nx.Graph: Copy of the full Zephyr graph with faulty nodes removed.
            All graph-level metadata (family, rows, tile, labels) is preserved.
    """
    rng = np.random.default_rng(seed)
    full_graph = zephyr_graph(m, t, coordinates=True)
    all_nodes = list(full_graph.nodes())
    N = len(all_nodes)

    # Phase 1: uniform random deletion
    n_uniform = round(proportion * uniform_proportion * N)
    uniform_indices = rng.choice(N, size=n_uniform, replace=False)
    deleted = {all_nodes[i] for i in uniform_indices}

    # Phase 2: iterative distance-based deletion with dynamic updates
    n_distance = round(proportion * (1 - uniform_proportion) * N)
    deleted_distance = set()

    if n_distance > 0 and deleted:
        # cumulative_dist[v] stores sum(dist(v, d) for d in current deleted set D)
        cumulative_dist = {node: 0.0 for node in all_nodes}
        for deleted_node in deleted:
            distances = nx.single_source_shortest_path_length(full_graph, deleted_node)
            for node, dist in distances.items():
                cumulative_dist[node] += dist

        for _ in range(n_distance):
            current_deleted = deleted | deleted_distance
            remaining = [node for node in all_nodes if node not in current_deleted]
            if not remaining:
                break

            weights = np.array(
                [
                    (1.0 / cumulative_dist[node]) if cumulative_dist[node] > 0 else 0.0
                    for node in remaining
                ]
            )
            total_weight = float(weights.sum())
            probs = weights / total_weight
            chosen_index = rng.choice(len(remaining), size=1, p=probs)[0]
            chosen_node = remaining[chosen_index]
            deleted_distance.add(chosen_node)

            distances = nx.single_source_shortest_path_length(full_graph, chosen_node)
            for node, dist in distances.items():
                cumulative_dist[node] += dist

    faulty_graph = full_graph.copy()
    faulty_graph.remove_nodes_from(deleted | deleted_distance)
    return faulty_graph


class TestYieldImprovement(unittest.TestCase):
    """Check that the greedy search never reduces the yield objective."""

    _SOURCE_M = 6
    _SOURCE_TP = 2
    _TARGET_M = 6
    _TARGET_T = 4
    _PROPORTION = 0.10
    _UNIFORM_PROPORTION = 0.10
    _SEED = 7795
    _TRUE_FALSE = [True, False]
    _YIELD_TYPES = ["node", "edge", "rail-edge"]
    _BY_STRATEGIES = ["by_quotient_rail", "by_quotient_node", "by_rail_then_node"]

    @classmethod
    def setUpClass(cls):
        cls.source = zephyr_graph(cls._SOURCE_M, cls._SOURCE_TP, coordinates=True)
        cls.target = generate_faulty_zephyr_graph(
            cls._TARGET_M,
            cls._TARGET_T,
            proportion=cls._PROPORTION,
            uniform_proportion=cls._UNIFORM_PROPORTION,
            seed=cls._SEED,
        )
        # Make sure that the target is a connected graph:
        if not nx.is_connected(cls.target):
            raise ValueError("Generated target graph is not connected; adjust parameters or seed.")

    def _assert_search_improves_yield(
        self, yield_type, quotient_search, expand_boundary_search, ksymmetric,
    ):
        sub_emb, metadata = zephyr_quotient_search(
            self.source,
            self.target,
            yield_type=yield_type,
            quotient_search=quotient_search,
            expand_boundary_search=expand_boundary_search,
            ksymmetric=ksymmetric,
        )

        self.assertIsInstance(metadata, ZephyrSearchMetadata)
        self.assertGreaterEqual(
            metadata.final_num_yielded,
            metadata.starting_num_yielded,
            msg=(
                f"Yield decreased from {metadata.starting_num_yielded} to "
                f"{metadata.final_num_yielded} with yield_type={yield_type}, "
                f"quotient_search={quotient_search}, "
                f"expand={expand_boundary_search}, ksymmetric={ksymmetric}"
            ),
        )
        # this should be impossible, but just double checking:
        self.assertLessEqual(metadata.final_num_yielded, metadata.max_num_yielded)

        target_nodes = set(self.target.nodes())
        # check the nodes the source was embedded onto are actually in the target
        # Flatten the chain tuples to check if all target nodes are in the target graph
        all_target_nodes = {node for chain in sub_emb.values() for node in chain}
        self.assertTrue(all_target_nodes.issubset(target_nodes))
        # check the nodes in the subgraph embedding are actually in the source
        self.assertTrue(set(sub_emb.keys()).issubset(set(self.source.nodes())))

    def test_search_yields_improvement(self):
        for quotient_search, expand, ksym, yt in itertools.product(
            self._BY_STRATEGIES, self._TRUE_FALSE, self._TRUE_FALSE, self._YIELD_TYPES,
        ):
            with self.subTest(
                quotient_search=quotient_search,
                expand_boundary_search=expand,
                ksymmetric=ksym,
                yield_type=yt,
            ):
                self._assert_search_improves_yield(
                    yield_type=yt,
                    quotient_search=quotient_search,
                    expand_boundary_search=expand,
                    ksymmetric=ksym,
                )


class TestMetadataConsistency(unittest.TestCase):
    """Verify the ZephyrSearchMetadata fields are internally consistent."""

    @classmethod
    def setUpClass(cls):
        cls.source = zephyr_graph(6, 2, coordinates=True)
        cls.target = generate_faulty_zephyr_graph(
            6, 4, proportion=0.10, uniform_proportion=0.10, seed=7795
        )

    def test_metadata_ordering(self):
        """max >= final >= starting >= 0 for all yield types."""
        for yt in ("node", "edge", "rail-edge"):
            with self.subTest(yield_type=yt):
                _sub, metadata = zephyr_quotient_search(
                    self.source,
                    self.target,
                    yield_type=yt,
                )
                self.assertGreaterEqual(metadata.max_num_yielded, 0)
                self.assertGreaterEqual(metadata.starting_num_yielded, 0)
                self.assertGreaterEqual(metadata.final_num_yielded, 0)
                self.assertGreaterEqual(
                    metadata.max_num_yielded, metadata.final_num_yielded
                )
                self.assertGreaterEqual(
                    metadata.final_num_yielded, metadata.starting_num_yielded
                )

    def test_full_target_gives_full_yield(self):
        """A perfect target should achieve full yield immediately (starting == final == max)."""
        full_target = zephyr_graph(6, 4, coordinates=True)
        for yt in ("node", "edge"):
            with self.subTest(yield_type=yt):
                _sub, metadata = zephyr_quotient_search(
                    self.source,
                    full_target,
                    yield_type=yt,
                )
                self.assertEqual(metadata.starting_num_yielded, metadata.max_num_yielded)
                self.assertEqual(metadata.final_num_yielded, metadata.max_num_yielded)

    def test_return_is_two_tuple(self):
        sub_emb, metadata = zephyr_quotient_search(self.source, self.target)
        self.assertIsInstance(sub_emb, dict)
        self.assertIsInstance(metadata, ZephyrSearchMetadata)


class TestGraphInputValidation(unittest.TestCase):
    """Tests for TypeError / ValueError raised by _validate_graph_inputs."""

    def setUp(self):
        self.source = zephyr_graph(6, 2, coordinates=True)
        self.target = zephyr_graph(6, 4, coordinates=True)

    def test_non_graph_source_or_target_raises_type_error(self):
        with self.assertRaisesRegex(TypeError, r"source and target must both be networkx"):
            zephyr_quotient_search("not_a_graph", self.target)  # type: ignore
        with self.assertRaisesRegex(TypeError, r"source and target must both be networkx"):
            zephyr_quotient_search(self.source, 42)  # type: ignore

    def test_source_or_target_wrong_family_raises_value_error(self):
        bad_graph = self.source.copy()
        bad_graph.graph["family"] = "chimera"
        with self.assertRaisesRegex(ValueError, r"source graph should be a zephyr family graph"):
            zephyr_quotient_search(bad_graph, self.target)
        with self.assertRaisesRegex(ValueError, r"target graph should be a zephyr family graph"):
            zephyr_quotient_search(self.source, bad_graph)

    def test_source_or_target_missing_rows_metadata_raises_value_error(self):
        graph_no_rows = self.source.copy()
        del graph_no_rows.graph["rows"]
        with self.assertRaisesRegex(ValueError, r"source graph is missing required 'rows'"):
            zephyr_quotient_search(graph_no_rows, self.target)
        with self.assertRaisesRegex(ValueError, r"target graph is missing required 'rows'"):
            zephyr_quotient_search(self.source, graph_no_rows)

    def test_source_or_target_missing_tile_metadata_raises_value_error(self):
        graph_no_tile = self.source.copy()
        del graph_no_tile.graph["tile"]
        with self.assertRaisesRegex(ValueError, r"source graph is missing required 'tile'"):
            zephyr_quotient_search(graph_no_tile, self.target)
        with self.assertRaisesRegex(ValueError, r"target graph is missing required 'tile'"):
            zephyr_quotient_search(self.source, graph_no_tile)

    def test_source_or_target_missing_labels_metadata_raises_value_error(self):
        graph_no_labels = self.source.copy()
        del graph_no_labels.graph["labels"]
        with self.assertRaisesRegex(ValueError, r"source graph is missing required 'labels'"):
            zephyr_quotient_search(graph_no_labels, self.target)
        with self.assertRaisesRegex(ValueError, r"target graph is missing required 'labels'"):
            zephyr_quotient_search(self.source, graph_no_labels)

    def test_incompatible_m_raises_value_error(self):
        target_diff_m = zephyr_graph(5, 4, coordinates=True)
        with self.assertRaisesRegex(
            ValueError, r"source and target must have the same number of rows"
        ):
            zephyr_quotient_search(self.source, target_diff_m)

    def test_target_tile_less_than_source_tile_raises_value_error(self):
        small_tile_target = self.target.copy()
        small_tile_target.graph["tile"] = 1  # less than source tp=2
        with self.assertRaisesRegex(
            ValueError, r"target tile count must be >= source tile count"
        ):
            zephyr_quotient_search(self.source, small_tile_target)

    def test_non_integer_rows_metadata_raises_type_error(self):
        bad_source = self.source.copy()
        bad_source.graph["rows"] = "six"
        with self.assertRaisesRegex(TypeError, r"graph 'rows' metadata must be an integer"):
            zephyr_quotient_search(bad_source, self.target)

    def test_non_positive_rows_metadata_raises_value_error(self):
        bad_source = self.source.copy()
        bad_source.graph["rows"] = 0
        with self.assertRaisesRegex(ValueError, r"graph 'rows' metadata must be positive"):
            zephyr_quotient_search(bad_source, self.target)


class TestSearchParameterValidation(unittest.TestCase):
    """Tests for TypeError / ValueError raised by _validate_search_parameters."""

    def setUp(self):
        self.source = zephyr_graph(6, 2, coordinates=True)
        self.target = zephyr_graph(6, 4, coordinates=True)

    def test_invalid_quotient_search_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, r"quotient_search must be one of"):
            zephyr_quotient_search(
                self.source, self.target, quotient_search="unknown_strategy"  # type: ignore
            )

    def test_invalid_yield_type_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, r"yield_type must be one of"):
            zephyr_quotient_search(
                self.source, self.target, yield_type="invalid"  # type: ignore
            )

    def test_non_dict_embedding_raises_type_error(self):
        with self.assertRaisesRegex(
            TypeError, r"embedding must be a dictionary when provided"
        ):
            zephyr_quotient_search(
                self.source, self.target, embedding=[1, 2, 3]  # type: ignore
            )

    def test_embedding_with_non_tuple_values_raises_type_error(self):
        """Embedding values must be tuples (chain format), not single nodes."""
        bad_embedding = {(0, 0, 0, 0, 0): [(0, 0, 0, 0, 0)]}  # List, not tuple
        with self.assertRaisesRegex(
            TypeError, r"embedding values must be tuples representing node chains"
        ):
            zephyr_quotient_search(
                self.source, self.target, embedding=bad_embedding  # type: ignore
            )

    def test_embedding_with_empty_chain_raises_value_error(self):
        """Embedding chains must be non-empty."""
        bad_embedding = {(0, 0, 0, 0, 0): ()}  # Empty chain
        with self.assertRaisesRegex(
            ValueError, r"embedding chains must be non-empty"
        ):
            zephyr_quotient_search(
                self.source, self.target, embedding=bad_embedding  # type: ignore
            )

    def test_embedding_with_non_5tuple_in_chain_raises_value_error(self):
        """Nodes in embedding chains must be 5-tuples."""
        bad_embedding = {(0, 0, 0, 0, 0): ((0, 0, 0, 0),)}  # 4-tuple instead of 5-tuple
        with self.assertRaisesRegex(
            ValueError, r"embedding chains must contain 5-tuples"
        ):
            zephyr_quotient_search(
                self.source, self.target, embedding=bad_embedding  # type: ignore
            )

    def test_embedding_with_duplicate_target_nodes_raises_value_error(self):
        """Embedding must be one-to-one: no duplicate target nodes across chains."""
        source_node1 = (0, 0, 0, 0, 0)
        source_node2 = (0, 0, 1, 0, 0)
        duplicate_target = (1, 1, 1, 1, 1)
        bad_embedding = {
            source_node1: (duplicate_target,),
            source_node2: (duplicate_target,),  # Duplicate target
        }
        with self.assertRaisesRegex(
            ValueError, r"embedding must be a one-to-one mapping.*duplicate target nodes"
        ):
            zephyr_quotient_search(
                self.source, self.target, embedding=bad_embedding  # type: ignore
            )

    def test_valid_chain_embedding_is_accepted(self):
        """Valid chain embedding with proper format should be accepted."""
        source = zephyr_graph(6, 2, coordinates=True)
        target = zephyr_graph(6, 4, coordinates=True)
        # Create a valid small chain embedding (identity mapping)
        valid_embedding = {node: (node,) for i, node in enumerate(source.nodes()) if i < 10}
        # Should not raise any errors
        try:
            zephyr_quotient_search(source, target, embedding=valid_embedding)
        except (TypeError, ValueError) as e:
            self.fail(f"Valid embedding raised unexpected error: {e}")


class TestLabelingSchemeErrors(unittest.TestCase):
    """Tests for ValueError raised by _ensure_coordinate_source / _ensure_coordinate_target."""

    def test_unknown_source_labels_raises_value_error(self):
        source = zephyr_graph(6, 2, coordinates=True)
        source.graph["labels"] = "custom_scheme"
        target = zephyr_graph(6, 4, coordinates=True)
        with self.assertRaisesRegex(ValueError, r"unknown labelling scheme"):
            zephyr_quotient_search(source, target)

    def test_unknown_target_labels_raises_value_error(self):
        source = zephyr_graph(6, 2, coordinates=True)
        target = zephyr_graph(6, 4, coordinates=True)
        target.graph["labels"] = "custom_scheme"
        with self.assertRaisesRegex(ValueError, r"unknown labelling scheme"):
            zephyr_quotient_search(source, target)
