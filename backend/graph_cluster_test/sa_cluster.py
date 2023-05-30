import networkx as nx
import numpy as np
from sklearn.preprocessing import LabelEncoder


def get_cluster(G, k=5):
    """
    :param G: 线图的 networkx 对象
    :param k: k-means 的簇数
    :return cluster_point_dict: map<graph_cluster_id, ['12_34', '34_56']>，值是数值，其中的元素是线图节点的名称
    """
    # adj matrix: links 邻接矩阵
    links = np.asarray(nx.to_numpy_matrix(G, dtype=np.float64))
    # attr 'value' of each id
    attributes = np.fromiter(nx.get_node_attributes(G, 'value').values(), dtype=int)
    # attributes = np.array(list(nx.get_node_attributes(G, 'value').values()))
    # attributes 中的值要求是从 0 开始的连续整数，而传进的 label 不符合要求，通过 LabelEncoder 的 fit_transform 实现映射
    print('值映射到从 0 开始的连续整数前', attributes)
    le = LabelEncoder()
    attributes = le.fit_transform(attributes)
    print('值映射到从 0 开始的连续整数后', attributes)
    attributes = attributes.reshape(attributes.shape[0], 1)

    # transition_probability_matrix and weights
    n = links.shape[0]  # n 个节点，todo：跟 G.nodes() 长度比较一下【确实是节点数】
    attrs = int(attributes.shape[1]) # 应该为 1【对，是每个节点内属性值的个数】
    w = np.ones(attrs + 1, dtype=float)  # 存储每个子属性的权重，todo：了解为什么要+1，是不是有另外存储什么信息

    # calculating the number of diff types of sub attr in a particular attribute
    # no_of_attr_individual 数组存储每个子属性中不同值的数量，而
    # no_of_attr_cumulative 数组存储每个子属性中不同值的累积数量
    no_of_attr_individual = np.zeros(attrs, dtype=float)
    no_of_attr_cumulative = np.zeros(attrs, dtype=float)
    for i in range(attrs):
        no_of_attr_individual[i] = np.unique(attributes[:, i]).shape[0]
        if i == 0:
            no_of_attr_cumulative[i] = no_of_attr_individual[i]
        else:
            no_of_attr_cumulative[i] = no_of_attr_cumulative[i - 1] + no_of_attr_individual[i]

    transition_pm = np.zeros((n + int(no_of_attr_cumulative[attrs - 1]), n + int(no_of_attr_cumulative[attrs - 1])),
                             dtype=float)
    c = 0.5
    l = 5

    def caltransition(w):
        # trasition matrix Pa
        # filling matrix of vertex to vertex
        sigma_w = np.sum(w) - w[0]
        for i in range(n):
            total_neighbours = np.sum(links[i])
            indices = np.where(links[i] == 1)
            transition_pm[i][indices] = w[0] / (total_neighbours * w[0] + sigma_w)

            for j in range(attrs):
                if j == 0:
                    transition_pm[i][n + int(attributes[i][j])] = w[j + 1] / (total_neighbours * w[0] + sigma_w)
                else:
                    transition_pm[i][n + int(attributes[i][j]) + no_of_attr_cumulative[j - 1]] = w[j + 1] / (
                                total_neighbours * w[0] + sigma_w)

        # filling matrix from attribute vertex to vertex
        for i in range(attrs):
            temp_neighbours_attr = np.zeros(int(no_of_attr_individual[i]))
            num_attr = int(no_of_attr_individual[i])
            for j in range(num_attr):
                temp_neighbours_attr[j] = np.where(attributes[:, i] == j)[0].size
                if i == 0:
                    transition_pm[n + j][np.where(attributes[:, i] == j)] = 1. / temp_neighbours_attr[j]
                else:
                    transition_pm[n + j + int(no_of_attr_cumulative[i - 1])][np.where(attributes[:, i] == j)] = 1. / \
                                                                                                            temp_neighbours_attr[
                                                                                                                j]
        # random walk

        random_walk = np.zeros((n + int(no_of_attr_cumulative[attrs - 1]), n + int(no_of_attr_cumulative[attrs - 1])),
                               dtype=float)
        for i in range(1, l + 1):
            random_walk += c * ((1 - c) ** i) * np.linalg.matrix_power(transition_pm, i)
        random_walk = random_walk[:n, :n]
        return random_walk

    random_walk = caltransition(w)

    # cluster
    sigma = 1
    # k = 5

    influence_function = 1 - np.exp(-np.square(random_walk) / (2 * (sigma ** 2)))
    density_function = np.sum(influence_function, axis=0)
    # doubt on adding at axis
    centroid = np.argsort(-density_function, kind='mergesort')[:k]  # centroid has indices of top k in desc

    curr_cluster = np.zeros(n, dtype='int')
    past_cluster = np.zeros(n, dtype='int')
    # cluster_averages = np.empty((k,n))

    unstable = True
    limit_step = 50

    while unstable and limit_step > 0:
        limit_step -= 1
        for i in range(n):
            curr_cluster[i] = np.argmax(random_walk[i, centroid])
        if np.array_equal(curr_cluster, past_cluster):
            unstable = False
        past_cluster = np.array(curr_cluster)

        for i in range(k):
            which_cluster = np.where(curr_cluster == i)[0]
            if which_cluster.size != 0:
                cluster_average = np.mean(random_walk[which_cluster, :], axis=0)
                cluster_avgpoint_dist = np.zeros(which_cluster.size)
                for j, node in enumerate(which_cluster):
                    cluster_avgpoint_dist[j] = np.linalg.norm(random_walk[node] - cluster_average)
                centroid[i] = which_cluster[np.argmin(cluster_avgpoint_dist)]

        dw = np.zeros(attrs, dtype=float)
        for i in range(k):
            which_cluster = np.where(curr_cluster == i)[0]
            if which_cluster.size != 0:
                for j in range(attrs):
                    dw[j] += np.sum(attributes[centroid[i]][j] == attributes[which_cluster][:, j:j + 1])
        sdw = np.sum(dw)
        for i in range(attrs):
            w[i + 1] = (w[i + 1] + (dw[i] / sdw)) / 2
        random_walk = caltransition(w)

        obj = 0.
        for i in range(k):
            which_cluster = np.where(curr_cluster == i)[0]
            if which_cluster.size != 0:
                obj += np.mean(random_walk[which_cluster, which_cluster])
            # print(which_cluster.size)

        print('obj', obj)
    print('curr', curr_cluster)

    cluster_point_dict = {}
    for i in range(k):
        cluster_point_dict[i] = []
        node_idxs_in_cluster = np.where(curr_cluster == i)[0].tolist()
        # 获取G的节点列表
        node_list = list(G.nodes())
        for node_id in node_idxs_in_cluster:
            node = node_list[node_id]
            if G.degree(node) == 0:
                continue
            node_name = G.nodes[node]['name'].split('-')
            node_name = f'{node_name[0]}_{node_name[1]}'
            cluster_point_dict[i].append(node_name)
            # print(G.nodes[node]['name'])
        # cluster_point_dict[i] = np.where(curr_cluster == i)[0].tolist()
    point_cluster_dict = curr_cluster.tolist()

    return point_cluster_dict, cluster_point_dict


def update_graph_with_attr(G, node_label_dict):
    print(G)
    for i, node in enumerate(G.nodes()):
        # print(f'{node[0]}_{node[1]}')
        if node_label_dict is None:
            label = i
        else:
            label = node_label_dict[f'{node[0]}_{node[1]}']
        G.nodes[node]['value'] = label
    # print('看看有没有value', G.nodes.data())
    return G


if __name__ == '__main__':
    G = nx.read_gml('polblogs.gml', label='id')

    # prints the data connections
    print('Nodes: ', G.number_of_nodes())
    print('Edges: ', G.number_of_edges())

    get_cluster(G, 4)
