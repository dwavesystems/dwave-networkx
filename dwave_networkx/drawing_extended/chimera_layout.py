from dwave_networkx import draw


def draw_chimera(G):
    draw(G, circular_layout(G), **kwargs)
