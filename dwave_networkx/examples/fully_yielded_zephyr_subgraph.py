# Copyright 2026 D-Wave
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

import networkx as nx
import numpy as np
from minorminer import find_embedding
from minorminer.utils.parallel_embeddings import find_sublattice_embeddings

import dwave_networkx as dnx
from dwave_networkx.generators._zephyr_playground import zephyr_quotient_search

seed = 12345
rng = np.random.default_rng(seed)

tile = dnx.zephyr_graph(6, 4, coordinates=True)
target = dnx.zephyr_graph(12, 4, coordinates=True)

#  first, identify one complete m=6, t=4 sublattice in the pristine target.
reference_embeddings = find_sublattice_embeddings(
    S=tile,
    T=target,
    max_num_emb=1,
    one_to_iterable=False,
    seed=seed,
)


# now, remove 10% random nodes from outside the sublattice that was found before
protected_nodes = set(reference_embeddings[0].values())
num_remove = int(0.1 * target.number_of_nodes())
removable_nodes = [n for n in target.nodes() if n not in protected_nodes]
removed_idx = rng.choice(len(removable_nodes), size=num_remove, replace=False)
removed_nodes = [removable_nodes[i] for i in removed_idx]
target.remove_nodes_from(removed_nodes)

# this finishes up creating our "defective" target graph, which, by construction, still contains at
# least one complete m=6, t=4 sublattice, but is now missing 10% of the nodes outside that
# sublattice.

# our example actually starts here. we start from this defective target graph, so we need to
# discover a complete m=6, t=4 sublattice in the defective target.
tile_embeddings = find_sublattice_embeddings(
    S=tile,
    T=target,
    max_num_emb=1,
    one_to_iterable=False,
    seed=seed,
)
tile_embedding = tile_embeddings[0]  # pick the first embedding.

# Relabel to canonical m=6 coordinates before zephyr_quotient_search.
sublattice_nodes = set(tile_embedding.values())
target_sub = target.subgraph(sublattice_nodes).copy()
inv_map = {target_node: tile_node for tile_node, target_node in tile_embedding.items()}
target_sub = nx.relabel_nodes(target_sub, inv_map, copy=True)
target_sub.graph.update(family="zephyr", rows=6, tile=4, labels="coordinate")

# embed source zephyr(mp=6, tp=2) into the found complete m=6, t=4 sublattice.
source = dnx.zephyr_graph(6, 2, coordinates=True)
emb, metadata = zephyr_quotient_search(source, target_sub, yield_type="edge")

print("quotient search yield:", metadata.final_num_yielded, "/", metadata.max_num_yielded)

# If not full-yield, refine with minorminer.find_embedding.
best_embedding = emb
if metadata.final_num_yielded < metadata.max_num_yielded:
    refined = find_embedding(
        S=source,
        T=target_sub,
        initial_chains=emb,
        timeout=50,
    )
    if refined:
        best_embedding = refined
else:
    print("full-yield embedding found with zephyr_quotient_search")

# mp back to original target labels, whichh can be used as the effective embedding for the source
# into the original target.
embedding_in_original_target = {
    s: tuple(tile_embedding[v] for v in chain)
    for s, chain in best_embedding.items()
}
