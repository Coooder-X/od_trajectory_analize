import networkx as nx  #导入networkx包
import matplotlib.pyplot as plt

from graph_process.Point import Point


class Graph:
    def __init__(self):
        self.G = nx.MultiDiGraph()
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
            L.nodes[node]['name'] = self.dict[start].name + '-' + self.dict[end].name
            curEdge = self.multiEdgeDict[(node[0], node[1])][-1]
            self.multiEdgeDict[(node[0], node[1])].pop()
            L.nodes[node]['Lnode_feature'] = curEdge  # origin_G[node[0]][node[1]] # 原图边的信息作为线图的点信息加入。但原图可能有多条等效边，因此线图需要依次为等效点分配feat，这里还没处理

        for edge in L.edges():  # like [((1, 3, 0), (3, 1, 0), 0), ((1, 3, 0), (3, 4, 0), 0)]，第3维恒是0
            # print('edge', edge) # like ((1, 3, 0), (3, 1, 0))
            # origin_edge1、origin_edge2 是线图中当前边的2端点，也是原图中相连的两条边，st 的结尾是end的开头
            st, end = edge[0], edge[1]  # like (1, 3, 0)
            # print('st end', st, end)
            connectPoint = st[1]  # 原图中2条邻接轨迹的连接点
            # print('connectPoint', connectPoint)
            # print(G.edges[(st, end, 0)])
            # print('connectPoint', self.dict[connectPoint])
            L.edges[(st, end, 0)]['Ledge_feature'] = self.dict[connectPoint].feature  # 'feat of origin point ' + str(connectPoint)

        # print('line_graph', L.nodes.data())
        # print(L.edges.data())
        return L

    def drawLineGraph(self):
        L = self.getLineGraph()
        print('线图中的点数', len(L.nodes))
        print('线图中的边数', len(L.edges))
        pos = nx.random_layout(L)
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

    g.drawGraph()
    g.drawLineGraph()


if __name__ == '__main__':
    main()