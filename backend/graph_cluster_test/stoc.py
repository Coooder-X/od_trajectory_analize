import bisect
import graph_tool as gt
from graph_tool.draw import graph_draw
import math
import queue
import random
from typing import List, Set


# An implementation of the Semantic-Topological Clustering (SToC) algorithm from "Efficiently Clustering Very Large Attributed Graphs"
# https://arxiv.org/pdf/1703.08590.pdf


def jaccard_distance(X: set, Y: set) -> float:
    return 1 - (len(X.intersection(Y)) / len(X.union(Y)))


def jaccard_feature(x: int, y: int) -> float:
    return jaccard_distance({x}, {y})


def categorical_feature(v, i, G: gt.GraphView):
    return G.vertex_properties[G.graph_properties['categorical_features'][i]][G.vertex(v)]


def quantitative_feature(v, i, G: gt.GraphView):
    return G.vertex_properties[G.graph_properties['quantitative_features'][i]][G.vertex(v)]


def distance_semantic(s: int, x: int, G: gt.GraphView) -> float:
    A = len(G.graph_properties['categorical_features']) + len(G.graph_properties['quantitative_features'])
    Q = len(G.graph_properties['quantitative_features'])
    return (math.sqrt(sum([(quantitative_feature(s, i, G) - quantitative_feature(x, i, G))**2 for i in range(Q)])) * math.sqrt(Q) + sum([jaccard_feature(categorical_feature(s, i, G), categorical_feature(x, i, G)) for i in range(A - Q)])) / A


def distance_topological(s: int, x: int, G: gt.GraphView) -> float:
    s_neighbors = G.vertex_properties['sketches'][G.vertex(s)]
    s_neighbors.add(s)
    x_neighbors = G.vertex_properties['sketches'][G.vertex(x)]
    x_neighbors.add(x)

    return jaccard_distance(s_neighbors, x_neighbors)


def distance(s: int, x: int, G: gt.GraphView) -> float:
    return max(distance_semantic(s, x, G), distance_topological(s, x, G))


def sto_query(G: gt.GraphView, tau: float, s: int) -> Set[int]:
    Q: "queue.Queue[int]" = queue.Queue()
    C: Set[int] = {s}
    Q.put(s)

    while not Q.empty():
        v: int = Q.get()
        for x in G.get_all_neighbors(v):
            if x not in C and distance(s, x, G) <= tau:
                C.add(x)
                Q.put(x)

    return C


def select_node(V: Set[int]) -> int:
    return random.sample(V, 1)[0]


def subgraph(G: gt.GraphView, vertices: Set[int]) -> gt.GraphView:
    v_filter = G.new_vertex_property('boolean')
    for v in G.get_vertices():
        if v in vertices:
            v_filter[G.vertex(v)] = True
        else:
            v_filter[G.vertex(v)] = False

    G.set_vertex_filter(v_filter)
    return G


def determine_k(n: int, epsilon: float) -> int:
    return round(math.log(n) / epsilon**2)


def calculate_bottom_k_sketches(G: gt.GraphView, k: int) -> gt.VertexPropertyMap:
    sketches = G.new_vertex_property('python::object')

    # assign a random order to the vertices
    # vertex v's rank will be given by vertex_rank[v]
    vertex_rank = [i for i in range(G.num_vertices())]
    random.shuffle(vertex_rank)

    for v in G.vertices():
        sketch = []
        ranks = []
        # todo generalize to use each vertex's l-neighborhood
        for neighbor in sorted(G.get_all_neighbors(v), key=lambda n: vertex_rank[n]):
            insert_pos = bisect.bisect(ranks, vertex_rank[neighbor])
            if len(ranks) < k or insert_pos < k:
                sketch.insert(insert_pos, neighbor)
                ranks.insert(insert_pos, vertex_rank[neighbor])
            else:
                break

        sketches[G.vertex(v)] = set(sketch[:k])

    return sketches


def stoc(G: gt.GraphView, tau: float, epsilon: float = 0.9) -> List[Set[int]]:
    sketches = calculate_bottom_k_sketches(G, determine_k(G.num_vertices(), epsilon))
    G.vertex_properties['sketches'] = sketches

    R: List[Set[int]] = []
    V_prime: Set[int] = set(G.get_vertices())

    while V_prime:
        s: int = select_node(V_prime)
        C: Set[int] = sto_query(G, tau, s)
        print(s, C)
        V_prime.difference_update(C)
        G = subgraph(G, V_prime)
        R.append(C)

    return R


def main():
    # Simple sample graph from the paper
    G = gt.Graph(directed=False)
    categorical_features = G.new_graph_property('vector<string>')
    G.graph_properties['categorical_features'] = categorical_features
    G.graph_properties['categorical_features'] = ['inhabited']
    quantitative_features = G.new_graph_property('vector<string>')
    G.graph_properties['quantitative_features'] = quantitative_features
    G.graph_properties['quantitative_features'] = ['x', 'y']

    G.add_vertex(8)
    label = G.new_vertex_property('string')
    pos = G.new_vertex_property('vector<float>')
    fill_color = G.new_vertex_property('vector<float>')

    inhabited = G.new_vertex_property('int')
    G.vertex_properties['inhabited'] = inhabited
    x = G.new_vertex_property('float')
    G.vertex_properties['x'] = x
    y = G.new_vertex_property('float')
    G.vertex_properties['y'] = y

    G.add_edge(0, 1)
    G.add_edge(0, 2)
    G.add_edge(0, 7)  # todo Can this edge be curved to avoid needing to manually adjust to have an uncluttered graph?
    G.add_edge(1, 2)
    G.add_edge(1, 3)
    G.add_edge(3, 4)
    G.add_edge(4, 5)
    G.add_edge(4, 6)
    G.add_edge(5, 6)
    G.add_edge(5, 7)
    G.add_edge(6, 7)

    inhabited[G.vertex(0)] = 1
    x[G.vertex(0)] = 0
    y[G.vertex(0)] = 0.1

    inhabited[G.vertex(1)] = 1
    x[G.vertex(1)] = 0
    y[G.vertex(1)] = 0

    inhabited[G.vertex(2)] = 1
    x[G.vertex(2)] = 0.1
    y[G.vertex(2)] = 0.1

    inhabited[G.vertex(3)] = 0
    x[G.vertex(3)] = 0.2
    y[G.vertex(3)] = 0

    inhabited[G.vertex(4)] = 0
    x[G.vertex(4)] = 0.4
    y[G.vertex(4)] = 0

    inhabited[G.vertex(5)] = 0
    x[G.vertex(5)] = 0.55
    y[G.vertex(5)] = 0.1

    inhabited[G.vertex(6)] = 0
    x[G.vertex(6)] = 0.6
    y[G.vertex(6)] = 0

    inhabited[G.vertex(7)] = 1
    x[G.vertex(7)] = 0.7
    y[G.vertex(7)] = 0.09

    colors = [
        [1, 0, 0, 1],  # red
        [1, .5, 0, 1],  # orange
        [1, 1, 0, 1],  # yellow
        [0, 1, 0, 1],  # green
        [0, 1, .5, 1],
        [0, 1, 1, 1],  # cyan
        [0, 0, 1, 1],  # blue
        [1, 0, 1, 1],  # purple
    ]

    for i in range(8):
        label[G.vertex(i)] = str(i)
        pos[G.vertex(i)] = [x[G.vertex(i)], -1*y[G.vertex(i)]]
        fill_color[G.vertex(i)] = colors[0] if inhabited[G.vertex(i)] else colors[6]

    graph_draw(G, pos=pos, vertex_text=label, vertex_fill_color=fill_color)

    tau: float = 0.6
    R = stoc(gt.GraphView(G), tau)
    print(R)

    i = 0
    for cluster in R:
        color = colors[i]
        for v in cluster:
            fill_color[G.vertex(v)] = color
        i += 1
    graph_draw(G, pos=pos, vertex_text=label, vertex_fill_color=fill_color)


if __name__ == '__main__':
    main()