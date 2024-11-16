import json
import os
import pickle
from datetime import datetime

# from cdlib import algorithms
import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
import torch

from graph_process.Graph import igraph2networkx, Graph
from graph_process.Point import Point

# from SpatialRegionTools import get_cell_id_center_coord_dict
# from args import args
# from gcc.graph_convolutional_clustering.gcc.run import run
# from graph_process.Point import Point
# from graph_process.Graph import get_degree_by_node_name, get_feature_list, get_adj_matrix, Graph, networkx2igraph, igraph2networkx
# import networkx as nx
#
# from od_graph_process import get_line_graph_by_selected_cluster
# from spatial_grid_utils import get_region, decode_od
# from t2vec import run_model2
# from visualization import vis_community

exp_log_name = 'exp_log'
exp_log = []

consider_edge_weight = True  # 是否考虑边权（在使用传统方法时有效）
use_line_graph = False  # 是否使用线图方法
use_igraph = False  # 无用，并始终为 False
tradition_method = 'CNM'  # 'Louvain' 'CNM'
draw_cluster = False  # 若执行线图方法，则可以选择是否绘制社区划分结果的图片

def CON(G, cluster_id, node_name_cluster_dict):
    start = datetime.now()
    m = len(G.edges())
    fz = 0
    for edge in G.edges():
        u, v = edge[0], edge[1]
        u_name, v_name = u, v
        if use_line_graph:
            u_name, v_name = f'{u[0]}_{u[1]}', f'{v[0]}_{v[1]}'
        if (node_name_cluster_dict[u_name] == cluster_id and node_name_cluster_dict[v_name] != cluster_id) or \
                (node_name_cluster_dict[u_name] != cluster_id and node_name_cluster_dict[v_name] == cluster_id):
            fz += 1
    vol_C = vol(G, cluster_id, node_name_cluster_dict)
    fm = fz + vol_C
    if fm == 0 or fz == 0:
        return -1
    print(f'分子={fz}， 分母={fm}')
    res = fz / fm
    return res


def vol(G, cluster_id, node_name_cluster_dict):
    res = 0
    for node in G.nodes():
        node_name = node
        if use_line_graph:
            node_name = f'{node[0]}_{node[1]}'
        if node_name in node_name_cluster_dict and node_name_cluster_dict[node_name] == cluster_id:
            res += G.degree(node)
    return res


def avg_CON(G, cluster_point_dict, node_name_cluster_dict, use_igraph):
    if use_igraph is True:
        G = igraph2networkx(G, nx.MultiDiGraph)
    avg = 0.0
    ok_cluster_num = 0
    for cluster_id in cluster_point_dict:
        cur_con = CON(G, cluster_id, node_name_cluster_dict)
        if cur_con == -1:
            continue
        ok_cluster_num += 1
        avg += cur_con
        print(f'cluster: {cluster_id} cur_con = {cur_con}')

    avg /= ok_cluster_num
    exp_log.append(f'cluster_num {len(cluster_point_dict.keys())} avg Con = {avg}')
    return avg


def get_ok_cluster_num(cluster_point_dict):
    ok_cluster_num = 0
    for cluster_id in cluster_point_dict:
        if len(cluster_point_dict[cluster_id]) > 5:
            ok_cluster_num += 1
    return ok_cluster_num


def get_origin_graph_by_selected_cluster(out_adj_dict, data_id):
    cluster_list = []  # 存储所有 Point 类型的 簇，作为 graph 的节点集
    cid_point_dict = {}  # 簇id 到 Point 类型的簇 的映射
    point_cid_dict = {}  # Point 类型的簇 到 簇id 的映射
    cluster_ids = set()
    for key in out_adj_dict:
        cluster_ids.add(key)
        for v in out_adj_dict[key]:
            cluster_ids.add(v)

    for cid in cluster_ids:
        point = Point(name=cid, nodeId=cid, infoObj={}, feature={})
        cluster_list.append(point)
        cid_point_dict[cid] = point
        point_cid_dict[point] = cid

    adj_point_dict = {}  # 根据 filtered_adj_dict 得出的等价的邻接表，索引是 Point 类型的簇
    for cid in out_adj_dict:
        point = cid_point_dict[cid]
        if point not in adj_point_dict:
            adj_point_dict[point] = []
        for to_cid in out_adj_dict[cid]:
            adj_point_dict[point].append(cid_point_dict[to_cid])

    od_flow_dict = {}
    with open(f"../data/selected_od_trj_dict_{data_id}.pkl", 'rb') as file:
        selected_od_trj_dict = pickle.loads(file.read())
    for key in selected_od_trj_dict:
        o, d = decode_od(key)
        od_flow_dict[(o, d)] = len(selected_od_trj_dict[key])

    g = Graph()
    for cluster in cluster_list:
        g.addVertex(cluster)
    #   边权值可以后续改成簇之间的 od 对数量，暂时默认为 1
    for point in adj_point_dict:
        edge = []
        u = point_cid_dict[point]
        for to_point in adj_point_dict[point]:
            v = point_cid_dict[to_point]
            if consider_edge_weight is True and (u, v) in od_flow_dict:
                edge.append([to_point, od_flow_dict[(u, v)]])
            else:
                edge.append([to_point, 1])
        g.addDirectLine(point, edge)
    return g, out_adj_dict


def decode_od(od, connect='_'):
    o, d = od.split(connect)
    return int(o), int(d)


# def get_grid_split(data_id):
#     out_adj_table = {}
#     with open(f"../json_data/od_graph_{data_id}.json") as json_g:
#         # 读取配置文件
#         od_list = json.load(json_g)
#     # od_list = [[od_pair['start'], od_pair['end'], ] for od_pair in od_list]
#     for od_pair in od_list:
#         if od_pair['start'] not in out_adj_table:
#             out_adj_table[od_pair['start']] = set()
#         out_adj_table[od_pair['start']].add(od_pair['end'])
#     return out_adj_table


# def get_line_graph(region, trj_region, out_adj_table, data_id):
#     with_space_dist = False
#     # 计算线图，返回适用于 d3 的结构和邻接表 ===========================
#     used_od_cells = set([x for x in range(100)])
#     tmp = {}
#     for start in out_adj_table:
#         if start in used_od_cells:
#             t = out_adj_table[start]
#             t = list(set(t).intersection(used_od_cells))
#             tmp[start] = t
#     out_adj_table = tmp
#     cid_center_coord_dict = get_cell_id_center_coord_dict(region)
#     selected_cluster_ids = set(cid_center_coord_dict.keys())
#     selected_cluster_ids = list(selected_cluster_ids.intersection(used_od_cells))
#     if use_line_graph is True:
#         g, filtered_adj_dict = get_origin_graph_by_selected_cluster(out_adj_table, data_id)
#         force_nodes, force_edges, line_graph_filtered_adj_dict, lg = get_line_graph_by_selected_cluster(
#             selected_cluster_ids, selected_cluster_ids, out_adj_table)
#         if use_igraph is True:
#             g = g.G
#             g_tmp = ig.Graph(directed=True)
#             g_tmp = g_tmp.from_networkx(g)
#             g = g_tmp.linegraph()
#         else:
#             g = lg
#     else:
#         g, filtered_adj_dict = get_origin_graph_by_selected_cluster(out_adj_table, data_id)
#         if use_igraph is True:
#             g = networkx2igraph(g.G)
#         else:
#             g = g.G
#
#     # print('边 ', lg.edges())
#     # print('点 ', lg.nodes())
#
#     node_names_trjFeats_dict = {}   # 节点名 -> 包含的轨迹特征数组的 map
#     if use_line_graph:
#         if os.path.isfile(args['best_model']):
#             print("=> loading best_model '{}'".format(args['best_model']))
#             best_model = torch.load(args['best_model'], map_location='cpu')
#         with open(f"../data/selected_od_trj_dict_{data_id}.pkl", 'rb') as file:
#             selected_od_trj_dict = pickle.loads(file.read())
#         for od_pair in selected_od_trj_dict.keys():
#             gps_trips = selected_od_trj_dict[od_pair]
#             gps_trips = [trip[2:] for trip in gps_trips]
#             features = get_trj_feats(gps_trips, best_model)
#             node_names_trjFeats_dict[od_pair] = features
#         print('node_names_trjFeats_dict', node_names_trjFeats_dict)
#
#     # ============== GCC 社区发现代码 ===============
#     if use_line_graph:
#         adj_mat = get_adj_matrix(g)  # 根据线图得到 csc稀疏矩阵类型的邻接矩阵
#         features, related_node_names = get_feature_list(lg, node_names_trjFeats_dict, 0)  # 根据线图节点顺序，整理一个节点向量数组，以及对应顺序的node name
#
#     # print(f'原图节点个数：{len(g.nodes())}')
#     # print('向量长度', len(features[0]))
#
#     ######## 仅在做实验时需要这个 for 循环，否则不需要循环，执行一次即可\
#     cluster_point_dict = {}
#     # for cluster_num in [10, 20, 30, 40, 50]:
#     for cluster_num in [4, 4, 5]:
#         weight = 'edge_feature' if consider_edge_weight is True else None
#         if not use_line_graph:
#             # louvain --------------------------------------------------------------------
#             if tradition_method == 'Louvain':
#                 communities = nx.algorithms.community.louvain_partitions(g, weight=weight, resolution=1.6, threshold=1e-06, seed=31)
#                 trj_labels = []
#                 for c in communities:
#                     trj_labels.append(c)
#                 print('trj==', trj_labels)
#                 communities = trj_labels[0]
#                 print('Louvain 社区发现结果为：', communities)
#                 node_name_cluster_dict = {}
#                 cluster_point_dict = {}
#                 for (i, cluster) in enumerate(communities):
#                     cluster_point_dict[i] = list(cluster)
#                     for cluster_id in cluster:
#                         node_name_cluster_dict[cluster_id] = i
#
#             # em --------------------------------------------------------------------------
#             # if tradition_method == 'em':
#             #     communities = algorithms.em(g, cluster_num)
#             #     communities = algorithms.async_fluid(g, cluster_num)
#             #     g_tmp = ig.Graph(directed=True)
#             #     g_tmp.add_vertices(list(g.nodes))
#             #     g_tmp.add_edges(list(g.edges))
#             #     g = g_tmp
#             #     communities = g.community_edge_betweenness(clusters=cluster_num, directed=True, weights=None)
#             #     print('=====> communities1=', communities)
#             #     trj_labels = communities
#             #     communities = list(communities.communities)
#             #     node_name_cluster_dict = {}
#             #     cluster_point_dict = {}
#             #     for (i, cluster) in enumerate(communities):
#             #         cluster_point_dict[i] = list(cluster)
#             #         for cluster_id in cluster:
#             #             node_name_cluster_dict[cluster_id] = i
#
#             # community_edge_betweenness (igraph)  ------------------------------------------------------
#             # if tradition_method == 'community_edge_betweenness':
#             #     com = g.community_edge_betweenness(clusters=cluster_num, directed=True, weights=None)
#             #     # com = g.community_leading_eigenvector(clusters=cluster_num, arpack_options=None, weights=None)
#             #     # print('com is ==============>', com)
#             #     # com = ig.GraphBase.community_edge_betweenness(g, 3, True)
#             #     print('com ===========>', com.as_clustering())
#             #     print('com ===========>', com)
#             #     communities = com.as_clustering()
#             #     trj_labels = communities
#             #     # communities = list(communities.communities)
#             #     node_name_cluster_dict = {}
#             #     cluster_point_dict = {}
#             #     for (i, cluster) in enumerate(communities):
#             #         cluster_point_dict[i] = list(cluster)
#             #         for cluster_id in cluster:
#             #             node_name_cluster_dict[cluster_id] = i
#
#             # asyn_lpa_communities --------------------------------------------------------
#             # if tradition_method == 'asyn_lpa_communities':
#             #     communities = nx.algorithms.community.asyn_lpa_communities(g, weight=weight, seed=None)
#             #     print('=====> communities1=', communities)
#             #     trj_labels = []
#             #     for c in communities:
#             #         trj_labels.append(c)
#             #     print('trj==', trj_labels)
#             #     communities = trj_labels
#             #     print('=====> communities2=', communities)
#             #     node_name_cluster_dict = {}
#             #     cluster_point_dict = {}
#             #     for (i, cluster) in enumerate(communities):
#             #         cluster_point_dict[i] = list(cluster)
#             #         for cluster_id in cluster:
#             #             node_name_cluster_dict[cluster_id] = i
#
#             # greedy_modularity_communities --------------------------------------------------------
#             if tradition_method == 'CNM':
#                 communities = nx.algorithms.community.greedy_modularity_communities(g, weight=weight, resolution=1.0, cutoff=1.8, best_n=None)
#                 trj_labels = []
#                 for c in communities:
#                     trj_labels.append(list(c))
#                 print('trj==', trj_labels)
#                 communities = trj_labels
#                 print('CNM 社区发现结果为：', communities)
#                 node_name_cluster_dict = {}
#                 cluster_point_dict = {}
#                 for (i, cluster) in enumerate(communities):
#                     cluster_point_dict[i] = list(cluster)
#                     for cluster_id in cluster:
#                         node_name_cluster_dict[cluster_id] = i
#
#         # 本文方法 ----------------------------------------------------------------------
#         if use_line_graph:
#             trj_labels = run(adj_mat, features, cluster_num)  # 得到社区划分结果，索引对应 features 的索引顺序，值是社区 id
#             trj_labels = trj_labels.numpy().tolist()
#             node_name_cluster_dict = {}
#             for i in range(len(trj_labels)):
#                 label = trj_labels[i]
#                 if label not in cluster_point_dict:
#                     cluster_point_dict[label] = []
#                 # 在线图中度为 0 的散点，视为噪声，从社区中排除
#                 # if get_degree_by_node_name(lg, related_node_names[i]) > 0:
#                 cluster_point_dict[label].append(related_node_names[i])
#                 node_name_cluster_dict[related_node_names[i]] = label
#             print('实际有效社区个数: ', len(cluster_point_dict.keys()))
#             exp_log.append(f'实际有效社区个数: {get_ok_cluster_num(cluster_point_dict)}')
#             print('cluster_point_dict', cluster_point_dict)
#             if draw_cluster:
#                 vis_community(cluster_point_dict, selected_od_trj_dict)
#
#         print(f'====> 社区个数：{cluster_num}, CON = {avg_CON(g, cluster_point_dict, node_name_cluster_dict, use_igraph)}')
#
#     file_name = f'../result/{exp_log_name}.txt'
#     f = open(file_name, 'w')
#     for log in exp_log:
#         f.write(log + '\n')
#     f.close()
#
#
# if __name__ == '__main__':
#     od_region = get_region()
#     data_id = 20  # 读取的.pkl文件的id
#     out_adj_table = get_grid_split(data_id)
#     trj_region = None
#     get_line_graph(od_region, trj_region, out_adj_table, data_id)
