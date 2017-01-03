"""Wraps the networkx Graph object"""

import networkx as nx
import json


class Graph(nx.Graph):
    def canonical_form(self):
        raise NotImplementedError('this is for dw_graph only, not yet implemented')

    def toJSON(self):
        raise NotImplementedError('this is for dw_graph only, not yet implemented')

if __name__ == '__main__':
    pass
