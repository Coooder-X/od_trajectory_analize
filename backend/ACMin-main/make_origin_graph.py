import json
import os
import pickle
from datetime import datetime

# from cdlib import algorithms
import igraph as ig
import numpy as np
import pandas as pd
import torch
import args
from cal_od import get_od_filter_by_day_and_hour, get_od_hot_cell, encode_od_point
from data_process import od_pair_process
from data_process.OD_area_graph import build_od_graph

from data_process.SpatialRegionTools import get_cell_id_center_coord_dict
from data_process.spatial_grid_utils import get_region, get_od_points_filter_by_region, divide_od_into_grid
# from exp5 import get_grid_split
# from SpatialRegionTools import get_cell_id_center_coord_dict
# from args import args
from gcc.graph_convolutional_clustering.gcc.run import run
from graph_process.Point import Point
from graph_process.Graph import get_degree_by_node_name, get_feature_list, get_adj_matrix, Graph, networkx2igraph, igraph2networkx
import networkx as nx

# from od_graph_process import get_line_graph_by_selected_cluster
# from spatial_grid_utils import get_region, decode_od
# from t2vec import run_model2
from scipy.sparse import csc_matrix, csr_matrix

from t2vec_graph import run_model2

# from visualization import vis_community

exp_log_name = 'exp_log'
exp_log = []

consider_edge_weight = True  # 是否考虑边权（在使用传统方法时有效）
use_line_graph = False  # 是否使用线图方法
use_igraph = False  # 无用，并始终为 False
tradition_method = 'CNM'  # 'Louvain' 'CNM'
draw_cluster = False  # 若执行线图方法，则可以选择是否绘制社区划分结果的图片

month = 5
start_day, end_day = 11, 12
start_hour, end_hour = 8, 10


def get_trj_feats(gps_trips, best_model):
    with open("../region.pkl", 'rb') as file:
        trj_region = pickle.loads(file.read())
    trj_feats = run_model2(args, gps_trips, best_model, trj_region)
    return trj_feats


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


def decode_od(od, connect='_'):
    o, d = od.split(connect)
    return int(o), int(d)


def get_origin_graph_by_selected_cluster(out_adj_dict, od_flow_dict):
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

    # od_flow_dict = {}
    # with open(f"../data/selected_od_trj_dict_{data_id}.pkl", 'rb') as file:
    #     selected_od_trj_dict = pickle.loads(file.read())
    # for key in selected_od_trj_dict:
    #     o, d = decode_od(key)
    #     od_flow_dict[(o, d)] = len(selected_od_trj_dict[key])

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


def get_origin_graph(region, out_adj_table, od_flow_dict):
    with_space_dist = False
    # 计算线图，返回适用于 d3 的结构和邻接表 ===========================
    used_od_cells = set([x for x in range(100)])
    tmp = {}
    for start in out_adj_table:
        if start in used_od_cells:
            t = out_adj_table[start]
            t = list(set(t).intersection(used_od_cells))
            tmp[start] = t
    out_adj_table = tmp
    cid_center_coord_dict = get_cell_id_center_coord_dict(region)
    selected_cluster_ids = set(cid_center_coord_dict.keys())
    selected_cluster_ids = list(selected_cluster_ids.intersection(used_od_cells))

    g, filtered_adj_dict = get_origin_graph_by_selected_cluster(out_adj_table, od_flow_dict)
    return g, filtered_adj_dict


def make_origin_graph(g):
    print(g.G.edges())
    file_name = './data/our/edgelist.txt'
    f = open(file_name, 'w')
    for edge in g.G.edges():
        f.write(f'{edge[0]} {edge[1]}\n')
    f.close()

    # 提取行索引、列索引和数据
    row_indices = []
    col_indices = []
    data = []

    for (row, col), [value] in g.multiEdgeDict.items():
        row_indices.append(row)
        col_indices.append(col)
        data.append(value)

    st = set(row_indices + col_indices)
    print('st', st)
    # 转换为 numpy 数组
    row_indices = np.array(row_indices)
    col_indices = np.array(col_indices)
    data = np.array(data)

    # 构造 csc_matrix
    sparse_matrix = csc_matrix((data, (row_indices, col_indices)), shape=(100, 100))
    print(sparse_matrix)
    with open('./data/our/attrs.pkl', 'wb') as file:
        pickle.dump(sparse_matrix, file)
    # with open('./data/our/attrs.pkl', 'rb') as file:
    #     features = pickle.load(file, encoding='latin1')
    # print(features)
    return


def get_grid_split(region, od_pair_set, hot_od_gps_set):
    #   研究区域确定、网格划分、轨迹数据的时间确定
    start_time = datetime.now()
    res = od_pair_process.get_od_points_filter_by_day_and_hour(month, start_day, end_day)
    print(f'start {start_day} end {end_day}')
    od_points = np.array(res['od_points'])
    total_od_coord_points = od_points[:, 0:2]  # 并去掉时间戳留下经纬度坐标
    print('读取OD点结束，用时: ', (datetime.now() - start_time))
    res = od_pair_process.get_od_points_filter_by_day_and_hour(month, start_day, end_day, start_hour, end_hour)
    index_lst = res['index_lst']
    part_od_points = res['od_points']
    tmp_index, tmp_od = [], []
    print('++++++++++++++++++', part_od_points[0])
    for i in range(len(part_od_points)):
        if encode_od_point(part_od_points[i]) in hot_od_gps_set:
            tmp_index.append(index_lst[i])
            tmp_od.append(part_od_points[i])
    part_od_points = tmp_od
    index_lst = tmp_index

    part_od_points, index_lst = get_od_points_filter_by_region(region, part_od_points, index_lst)
    point_cluster_dict, cluster_point_dict = divide_od_into_grid(region, part_od_points, index_lst)
    out_adj_table, in_adj_table = build_od_graph(point_cluster_dict, od_points, index_lst)

    return {
        'index_lst': index_lst,  # 当前小时时间段内的部分 OD 点索引
        'point_cluster_dict': point_cluster_dict,  # 全量的
        'cluster_point_dict': cluster_point_dict,  # 全量的
        'part_cluster_point_dict': cluster_point_dict,  # 当前小时内部分的映射关系，保证每个簇内的点都在当前小时段内
        'part_od_points': part_od_points,  # 当前小时段内部分的 OD 点
        'out_adj_table': out_adj_table,  # 当前小时段内过滤处的出边邻接表
        'in_adj_table': in_adj_table,  # 当前小时段内过滤处的入边邻接表
    }


if __name__ == '__main__':
    # od_region = get_region()
    # data_id = 20  # 读取的.pkl文件的id
    # out_adj_table = get_grid_split(data_id)
    # trj_region = None
    # g, filtered_adj_dict = get_origin_graph(od_region, trj_region, out_adj_table, data_id)
    # make_origin_graph(g, filtered_adj_dict)
    od_region = get_region()
    # makeVocab(trj_region, h5_files)
    total_od_pairs = get_od_filter_by_day_and_hour(month, start_day, end_day, start_hour, end_hour, od_region)

    # get_od_hot_cell 的后2个参数：1000 是只考虑当前区域和时间段内最热门的k=1000个OD对，lower_bound=0是过滤阈值，即流量大于0的OD都会被加入数据集
    od_pairs, od_cell_set, od_pair_set, hot_od_gps_set, od_flow_dict = get_od_hot_cell(total_od_pairs, od_region, 1000, 1)
    print('--------------> od pair', od_pairs, '\nhot_od_gps_set', hot_od_gps_set)
    res = get_grid_split(od_region, od_pair_set, hot_od_gps_set)
    print('res', res)
    out_adj_table = res['out_adj_table']
    print('out_adj_table', out_adj_table)
    g, filtered_adj_dict = get_origin_graph(od_region, out_adj_table, od_flow_dict)
    print('g', g)
    print('filtered_adj_dict', filtered_adj_dict)
    make_origin_graph(g)
    # get_line_graph(od_region, trj_region, month, start_day, end_day, start_hour, end_hour, res['out_adj_table'],
    #                res['cluster_point_dict'], od_pair_set)
