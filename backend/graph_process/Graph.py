import networkx as nx  # 导入networkx包
import matplotlib.pyplot as plt
import numpy as np
import igraph as ig

from graph_process.Point import Point


class Graph:
    def __init__(self):
        self.G = nx.MultiDiGraph()
        # self.G = nx.Graph()
        self.dict = {}  # 结点对应节点ID的字典
        self.nodeList = []
        self.multiEdgeDict = {}  # 存储旧图的重边的edge_feature，键为旧图的节点id构成的元组：(old_from_id, old_to_id)

    # 添加节点
    def addVertex(self, node):
        self.nodeList.append(node)
        self.dict[node.nodeId] = node
        self.G.add_node(node.nodeId, node_feature=node.feature)

    # 添加有向边
    def addDirectLine(self, start, ends):
        for row in ends:  # 遍历终点数组
            endNode = row[0]
            edge_info = row[1]  # 暂时是weight，之后可替换为特征向量
            self.G.add_edge(start.nodeId, endNode.nodeId, edge_feature=edge_info)
            # 保存重边信息，start 到 end 的边若没添加过，dict初始化为[]并加入边信息，否则添加边信息
            if (start.nodeId, endNode.nodeId) in self.multiEdgeDict:
                self.multiEdgeDict[(start.nodeId, endNode.nodeId)].append(edge_info)
            else:
                self.multiEdgeDict[(start.nodeId, endNode.nodeId)] = [edge_info]

    def drawGraph(self):
        G = self.G
        edges = G.edges()
        # print('edges', edges)

        # 生成节点标签
        labels = {}
        for nodeId in G.nodes:
            labels[nodeId] = self.dict[nodeId].name
        # 生成节点位置
        pos = nx.random_layout(G)  # circular_layout
        # pos = nx.circular_layout(G)  # circular_layout
        # 画节点
        nx.draw_networkx_nodes(G, pos, node_color='g', node_size=50, alpha=0.8)
        # 画边
        nx.draw_networkx_edges(G, pos, width=[float(v['edge_feature']) for (r, c, v) in G.edges(data=True)], alpha=0.5,
                               edge_color='b', connectionstyle='arc3, rad = 0.2')
        # 把边权重画出来
        edge_labels = dict([((u, v,), d['edge_feature']) for u, v, d in G.edges(data=True)])
        # print('weight of all edges:', edge_labels)
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.3, font_size=7)

        # 画节点的标签
        # nx.draw_networkx_labels(G, pos, labels, font_size=16)
        plt.figure(figsize=(70, 70))
        plt.axis('on')
        # 去掉坐标刻度
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def getLineGraph(self):
        origin_G = self.G
        # print('getLineGraph getGraph', origin_G)
        # print('origin_G', origin_G.nodes.data())
        # print('origin_G edges', origin_G.edges())
        # print('origin_G edges data', origin_G.edges.data())
        L = nx.line_graph(origin_G)
        # print('L.nodes', L.nodes())
        # print('L.nodes data', L.nodes.data())
        # print('L.edges', L.edges)
        for node in L.nodes():  # node like (1, 3, 0)
            # print('origin edge', origin_G[node[0]][node[1]])
            # print('origin edge', self.multiEdgeDict[(node[0], node[1])])
            start, end = node[0], node[1]
            L.nodes[node]['from'] = origin_G.nodes[start]
            L.nodes[node]['to'] = origin_G.nodes[end]
            L.nodes[node]['name'] = str(self.dict[start].name) + '-' + str(self.dict[end].name)
            # print(self.multiEdgeDict[(node[0], node[1])])
            #  TODO: 看一下 self.multiEdgeDict[(node[0], node[1])] 的长度什么情况会是 0，为 0 是不是正常的
            # if len(self.multiEdgeDict[(node[0], node[1])]) > 0:
            #     curEdge = self.multiEdgeDict[(node[0], node[1])][-1]
            #     self.multiEdgeDict[(node[0], node[1])].pop()
            #     L.nodes[node]['Lnode_feature'] = curEdge  # origin_G[node[0]][node[1]] # 原图边的信息作为线图的点信息加入。但原图可能有多条等效边，因此线图需要依次为等效点分配feat，这里还没处理

        for edge in L.edges():  # like [((1, 3, 0), (3, 1, 0), 0), ((1, 3, 0), (3, 4, 0), 0)]，第3维恒是0
            # print('edge', edge) # like ((1, 3, 0), (3, 1, 0))
            # origin_edge1、origin_edge2 是线图中当前边的2端点，也是原图中相连的两条边，st 的结尾是end的开头
            st, end = edge[0], edge[1]  # like (1, 3, 0)
            # print('st end', st, end)
            connectPoint = st[1]  # 原图中2条邻接轨迹的连接点
            # print('connectPoint', connectPoint)
            # print(G.edges[(st, end, 0)])
            # print('connectPoint', self.dict[connectPoint])
            # L.edges[(st, end, 0)]['Ledge_feature'] = self.dict[connectPoint].feature  # 'feat of origin point ' + str(connectPoint)

        # print('line_graph', L.nodes.data())
        # print(L.edges.data())
        return L

    def drawLineGraph(self):
        L = self.getLineGraph()
        print('线图中的点数', len(L.nodes))
        print('线图中的边数', len(L.edges))
        pos = nx.random_layout(L)
        # pos = nx.circular_layout(L)  # circular_layout
        # 生成节点标签
        labels = {}
        for node in L.nodes:
            labels[node] = L.nodes[node]['name']
        # 点label
        # nx.draw_networkx_labels(L, pos, labels, font_size=12)
        # 点
        nx.draw_networkx_nodes(L, pos, node_color='b', node_size=30, alpha=0.8)
        # 边
        nx.draw_networkx_edges(L, pos, alpha=0.5,
                               edge_color='g', connectionstyle='arc3, rad = 0.5')
        # 边label
        edge_labels = dict([((u, v,), d['Ledge_feature']) for u, v, d in L.edges(data=True)])
        # print('weight of all edges:', edge_labels)
        # nx.draw_networkx_edge_labels(L, pos, edge_labels=edge_labels, label_pos=0.3, font_size=7)

        # nx.draw(L,pos,connectionstyle='arc3, rad = 0.2')
        plt.axis('on')
        # 去掉坐标刻度
        plt.xticks([])
        plt.yticks([])
        plt.show()


def get_degree_by_node_name(G, name):
    """
    根据节点名称，查找 G 中该节点的度数。
    传入的 name 的形式是 {}_{}，而 G 中节点名称形式是 {}-{}，需要转换
    """
    src, tgt = name.split('_')
    for node in G.nodes:
        name1 = f'{src}-{tgt}'
        name2 = f'{tgt}-{src}'
        if G.nodes[node]['name'] == name1 or G.nodes[node]['name'] == name2:
            return G.degree(node)
    print('没找到节点')
    return 0

def get_adj_matrix(G):
    """
    得到的邻接矩阵的节点顺序，就是 G.nodes() 中节点的顺序，因此可以和下方 get_feature_list() 函数得到的 features 数组顺序对应
    """
    # mat = nx.adjacency_matrix(self.G)
    # A = np.array(nx.adjacency_matrix(G).todense())
    # print(mat)
    adj_mat = nx.to_scipy_sparse_array(G, format='csc')
    return adj_mat


def get_feature_list(G, node_feature_dict, avg_trj_num):
    """
    根据字典的键，即节点名称，整理出 G.nodes() 顺序的 节点特征数组
    字典的键 的形式是 {}_{}，而 G 中节点名称形式是 {}-{}，需要转换
    """
    features = []
    node_names = []
    shape = [768]
    for node in G.nodes:
        name = G.nodes[node]['name']
        src, tgt = name.split('-')
        name = f'{src}_{tgt}'
        # if name not in node_feature_dict:
        #     name = f'{tgt}_{src}'
        # feat = np.array(node_feature_dict[name])
        if name in node_feature_dict:
            node_feat = node_feature_dict[name]
            # print('==============> shape', node_feat[0].shape)
            if shape == [768]:
                shape = node_feat[0].shape
        else:
            node_feat = [np.zeros(shape)]
        feat = aggregate_node_trj_feat(node_feat, avg_trj_num)
        features.append(feat)
        node_names.append(name)
    features = np.array(features)
    return features, node_names


def aggregate_node_trj_feat(feat_lst, num):
    feat_mat = [np.array(feat) for feat in feat_lst]
    feat_mat = np.array(feat_mat)
    return np.average(feat_mat, axis=0)
    # feat_mat = []
    # for feat in feat_lst:
    #     feat_mat.extend(feat)
    #     # print('trj len', len(feat))
    # feat_mat = np.array(feat_mat)
    # # print(feat_mat)
    # # print('od len', len(feat_mat))
    # return feat_mat
    # feat_mat = []
    # for i in range(min(len(feat_lst), num)):
    #     feat_mat.extend(feat_lst[i])
    # while(len(feat_mat) < num):
    #     feat_mat.extend(np.zeros(feat_lst[0].shape))
    #
    # feat_mat = np.array(feat_mat)
    # return feat_mat


def get_dag_from_community(cluster_point_dict: dict, lg_force_edge: list):
    # 逻辑有误！！！
    dag_force_node = []
    dag_force_edge = []
    dag_node_names = list(cluster_point_dict.keys())
    edge_set = set([edge['name'] for edge in lg_force_edge])
    print('edge_set', edge_set)

    for cid in dag_node_names:
        dag_force_node.append({'name': f'{cid}'})
    for i in range(len(dag_node_names)):
        cid1 = dag_node_names[i]
        for j in range(i+1, len(dag_node_names)):
            cid2 = dag_node_names[j]
            edge_name1 = f'{cid1}_{cid2}'
            edge_name2 = f'{cid2}_{cid1}'
            if edge_name1 in edge_set:
                dag_force_edge.append({'source': cid1, 'target': cid2})
            if edge_name2 in edge_set:
                dag_force_edge.append({'source': cid2, 'target': cid1})

    print('dag_force_edge', dag_force_edge)
    print('dag_force_node', dag_force_node)
    return dag_force_node, dag_force_edge


def networkx2igraph(graph, directed=True):
    g = ig.Graph(directed=directed)
    g = g.from_networkx(graph)
    return g


def igraph2networkx(graph, graphClass):
    # g = graph.to_networkx(graph, graphClass)
    # return g
    edges = graph.get_edgelist()
    g = graphClass(edges)
    return g


def main():
    lst = [Point('1', 1, 1, {}), Point('2', 2, 2, {}), Point('3', 3, 3, {}),
           Point('4', 4, 4, {}), Point('5', 5, 5, {})]

    g = Graph()
    for k in lst:
        g.addVertex(k)

    g.addDirectLine(lst[0], [[lst[2], 2], [lst[2], 2.5], [lst[1], 1]])
    g.addDirectLine(lst[2], [[lst[3], 0.5], [lst[0], 3]])
    g.addDirectLine(lst[3], [[lst[4], 0.5]])
    g.addDirectLine(lst[4], [[lst[1], 4], [lst[3], 0.7]])

    print(g.G.edges())
    print(g.G.nodes())
    print(g.G.nodes)
    for a in g.G.nodes():
        for b in g.G.nodes():
            if (a, b) in g.G.edges():
                print(a, b)
                print((a, b))

    # print(g.G.nodes)
    # print()
    # for (i, node) in enumerate(g.G.nodes()):
    #     print(g.G.nodes[i])
        # print(g.G.nodes[node])
        # print(node)
    # g.drawGraph()
    # g.drawLineGraph()

    # lst = [Point('1', 1, 1, {}), Point('2', 2, 2, {}), Point('3', 3, 3, {}),
    #        Point('4', 4, 4, {})]
    #
    # g = Graph()
    # for k in lst:
    #     g.addVertex(k)
    #
    # g.addDirectLine(lst[0], [[lst[1], 2], [lst[1], 1]])
    # g.addDirectLine(lst[1], [[lst[0], 3]])
    # g.addDirectLine(lst[2], [[lst[0], 0.5], [lst[1], 1]])
    # g.addDirectLine(lst[3], [[lst[2], 4]])
    #
    # # g.drawGraph()
    # g.drawLineGraph()


if __name__ == '__main__':
    main()
