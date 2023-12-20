import os
import pickle
from datetime import datetime
import time
import threading

import numpy as np
import pandas as pd
import torch

import utils
from graph_process.Graph import Graph
from graph_process.Point import Point
from cal_od import exp_od_pair_set, get_od_filter_by_day_and_hour, get_od_hot_cell, encode_od_point
from data_process import od_pair_process
from data_process.OD_area_graph import build_od_graph, fuse_fake_edge_into_linegraph, \
    get_line_graph_by_selected_cluster
from data_process.SpatialRegionTools import get_cell_id_center_coord_dict, makeVocab, inregionS
from data_process.od_pair_process import get_trips_by_ids, get_trj_ids_by_force_node, \
    get_odpair_space_similarity
from data_process.spatial_grid_utils import get_region, get_od_points_filter_by_region, divide_od_into_grid
from gcc.graph_convolutional_clustering.gcc.run import run, draw_cluster_in_trj_view, draw_cluster_in_trj_view_new, draw_cluster_in_trj_view_new_exp4
from graph_process.Graph import get_degree_by_node_name, get_feature_list, get_adj_matrix
from t2vec import args
from t2vec_graph import run_model2, get_cluster_by_trj_feature
import networkx as nx

exp4_log_name = 'exp4_log'
exp4_log = []

def Q(G, node_name_cluster_dict):
    """
    @node_name_cluster_dict: 节点名到社区id的映射
    """
    node_names = set(list(node_name_cluster_dict.keys()))
    m = len(G.edges())
    res = 0.0
    for a in G.nodes():
        src, tgt = G.nodes[a]['name'].split('-')
        a_name = f'{src}_{tgt}'
        for b in G.nodes():
            # if a == b:
            #     continue
            src, tgt = G.nodes[b]['name'].split('-')
            b_name = f'{src}_{tgt}'
            Aab = 1 if (a, b, 0) in G.edges() else 0
            ksy = 0
            if a_name in node_names and b_name in node_names:
                ksy = 1 if node_name_cluster_dict[a_name] == node_name_cluster_dict[b_name] else 0
            # E = G.out_degree(a) * G.out_degree(b) / m
            # E = G.degree(a) * G.degree(b) / (2 * m)
            E = G.out_degree(a) * G.in_degree(b) / m
            res += (Aab - E) * ksy / m
            if (Aab - E) * ksy / m > 0:
                print(f'Aab={Aab}, E={E}, ksy={ksy}, cur={(Aab - E) * ksy / m}')
    exp4_log.append(f'cluster_num {len(set(list(node_name_cluster_dict.values())))} Q = {res}')
    # labels = list(node_name_cluster_dict.values())
    # for cluster_id in labels:
    #     ai = 0
    #     eij = 0
    #     for a in G.nodes():
    #         src, tgt = G.nodes[a]['name'].split('-')
    #         a_name = f'{src}_{tgt}'
    #         ksy = 1 if node_name_cluster_dict[a_name] == cluster_id else 0
    #         ai += (1 / m) * G.out_degree(a) * ksy
    #         for b in G.nodes():
    #             src, tgt = G.nodes[b]['name'].split('-')
    #             b_name = f'{src}_{tgt}'
    #             Aab = 1 if (a, b, 0) in G.edges() else 0
    #             ksy = node_name_cluster_dict[a_name] == cluster_id and node_name_cluster_dict[b_name] == cluster_id
    #             eij += (1 / m) * Aab * ksy
    #     res += eij - ai ** 2

    return res


def CON(G, cluster_id, node_name_cluster_dict):
    start = datetime.now()
    m = len(G.edges())
    fz = 0
    for edge in G.edges():
        u, v = edge[0], edge[1]
        # u_name, v_name = f'{u[0]}_{u[1]}', f'{v[0]}_{v[1]}'
        u_name, v_name = (u[0], u[1]), (v[0], v[1])
        if (node_name_cluster_dict[u_name] == cluster_id and node_name_cluster_dict[v_name] != cluster_id) or \
            (node_name_cluster_dict[u_name] != cluster_id and node_name_cluster_dict[v_name] == cluster_id):
            fz += 1
    vol_C = vol(G, cluster_id, node_name_cluster_dict)
    print(f'vol_C={vol_C}({cluster_id})')
    fm = fz + vol_C
    # fm = min(vol_C, m - vol_C)
    if fm == 0 or fz == 0:
        return -1
    end = datetime.now()
    print('用时', end - start)
    print(f'分子={fz}， 分母={fm}')
    # res = fz / (fz + vol_C + 0.01)
    # print(f'CON=({res})')
    res = fz / fm
    return res


def vol(G, cluster_id, node_name_cluster_dict):
    res = 0
    for node in G.nodes():
        src, tgt = G.nodes[node]['name'].split('-')
        # name = f'{src}_{tgt}'
        src, tgt = int(src), int(tgt)
        name = (src, tgt)
        if name in node_name_cluster_dict and node_name_cluster_dict[name] == cluster_id:
            res += G.degree(node)
    return res


def avg_CON(G, cluster_point_dict, node_name_cluster_dict):
    avg = 0.0
    print(len(cluster_point_dict.keys()))
    ok_cluster_num = 0
    for cluster_id in cluster_point_dict:
        # if len(cluster_point_dict[cluster_id]) > 5:
        cur_con = CON(G, cluster_id, node_name_cluster_dict)
        if cur_con == -1:
            continue
        ok_cluster_num += 1
        avg += cur_con
        print(f'cluster: {cluster_id} cur_con = {cur_con}')

    avg /= ok_cluster_num
    exp4_log.append(f'cluster_num {len(cluster_point_dict.keys())} avg Con = {avg}')
    return avg


def TPR(G, cluster_id, cluster_point_dict, name_node_dict):
    cluster = cluster_point_dict[cluster_id]    # 一个存储 node name 的数组
    cnt = 0
    for name_i in cluster:
        vi = name_node_dict[name_i]
        for name_j in cluster:
            if name_j == name_i:
                continue
            vj = name_node_dict[name_j]
            if (vi, vj, 0) not in G.edges():
                continue
            for name_w in cluster:
                if name_w == name_i or name_w == name_j:
                    continue
            vw = name_node_dict[name_w]
            if (vi, vj, 0) in G.edges() and (vj, vw, 0) in G.edges() and (vi, vw, 0) in G.edges():
                print('三个点 ', vi, vj, vw)
                print('三个点名称 ', name_i, name_j, name_w)
                cnt += 1

    edge_num = 0
    for name1 in cluster:
        node1 = name_node_dict[name1]
        for name2 in cluster:
            node2 = name_node_dict[name2]
            if (node1, node2, 0) in G.edges():
                edge_num += 1

    print(f'cluster = {cluster}')
    print(f'cnt = {cnt}, edge num = {edge_num}')
    if edge_num == 0:
        return 0

    print(f'TPR={cnt}/{edge_num} = {cnt / edge_num}')
    return cnt / edge_num


def get_ok_cluster_num(cluster_point_dict):
    ok_cluster_num = 0
    for cluster_id in cluster_point_dict:
        if len(cluster_point_dict[cluster_id]) > 5:
            ok_cluster_num += 1
    return ok_cluster_num


month = 5
start_day, end_day = 11, 12
start_hour, end_hour = 18, 20
# start_hour, end_hour = 8, 10

def get_grid_split(region):
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


def get_od_mat_by_time(day, hour):
    """
    :day    预测的那天的日期
    :hour   预测当天的第几小时,建议选择小数,如 6.2,因为时间片是 15min
    :return 返回的是一个numpy数组,维度是 1001
    """
    # data = np.load('./total_pred_100.npz', allow_pickle=True)['data']
    # # 一个时间片是 15min, 1/4 小时,所以是 *4
    # # data 中的数据是从第一周结束开始的,即 7*24*4=672 个时间片,因此
    # time_slice = (day - 1) * 24 * 4 + hour * 4 - (7 * 24 * 4)
    # # 维度是 1001, 去掉最后一个元素(TODO：不知道第一个还是最后一个是最大的，还是都不是，应该去掉小的哪个)
    # res = data[time_slice][:-1]
    # # print(res)
    # return res
    data = np.load('./history_od_data.npz', allow_pickle=True)['data']
    with open("./selet.pkl", 'rb') as file:
        selet = pickle.loads(file.read())['selet']
    time_slice = (day - 1) * 24 * 4 + hour * 4 - (7 * 24 * 4)
    res = data[time_slice][selet][:-1]
    return res


def get_od_mat_by_time_slice(time_slice):
    data = np.load('./history_od_data.npz', allow_pickle=True)['data']
    # with open("./selet.pkl", 'rb') as file:
    #     selet = pickle.loads(file.read())['selet']
    # res = data[time_slice][selet][:-1]
    # 维度是 1001, 去掉最后一个元素(TODO：不知道第一个还是最后一个是最大的，还是都不是，应该去掉小的哪个)
    res = data[time_slice][:-1]
    # print(res)
    return res


def build_line_graph_by_od_matrix(od_mat):
    print('od_mat', od_mat)
    with open("./selet.pkl", 'rb') as file:
        selet = pickle.loads(file.read())['selet']
    #  得到 top1000 流量的 OD 对, 记录下 OD 对索引和 OD 区域的索引
    # for i, s in enumerate(selet):
    #     print(f"{i}: {s}")
    top_od_pair = set()
    top_od_pair_idx = []
    for i in range(len(selet)):
        if selet[i] == True and len(top_od_pair) < 1000:
            o_id, d_id = i // 100, i % 100
            # d_id, o_id = i // 100, i % 100
            top_od_pair_idx.append(i)
            top_od_pair.add(f'{o_id}_{d_id}')  # 将索引转换为矩阵坐标,也就是OD区域的编号, 第一个是O,第二个是D(TODO 待考证)
    print('len------------======>', len(top_od_pair))
    #  从当前时间片的 OD 矩阵中找到这些 OD对, 将流量大于0的拿出来构建图
    od_pair = []
    od_flow_dict = {}
    od_idx_dict = {}  # 对于10000个OD对的索引是key, 值是key对应的在top1000中的索引
    for (i, od_pair_idx) in enumerate(top_od_pair_idx):
        o_id, d_id = od_pair_idx // 100, od_pair_idx % 100
        # d_id, o_id = od_pair_idx // 100, od_pair_idx % 100
        if od_mat[i] > 0 and f'{o_id}_{d_id}' in top_od_pair:
            # print('OD 对 ====>', [o_id, d_id])
            od_flow = float(od_mat[i])
            od_idx_dict[od_pair_idx] = i
            od_flow_dict[(o_id, d_id)] = [od_flow]
            od_pair.append([o_id, d_id])  # 将索引转换为矩阵坐标,也就是OD区域的编号, 第一个是O,第二个是D(TODO 待考证)

    out_adj_table = {}
    od_area_ids = set()  # 所有出现锅的 OD 区域 id
    for pair in od_pair:
        o_id, d_id = pair[0], pair[1]
        od_area_ids.add(o_id)
        od_area_ids.add(d_id)
        if o_id not in out_adj_table:
            out_adj_table[o_id] = []
        out_adj_table[o_id].append(d_id)

    # print('out_adj_table', out_adj_table)
    print('od_area_ids', od_area_ids)

    point_list = []  # 存储所有 point 对象的数组
    od_id_point_dict = {}  # 存储 OD 区域 id 到 point 对象的映射
    for od_id in od_area_ids:
        point = Point(name=od_id, nodeId=od_id, infoObj={}, feature={})
        point_list.append(point)
        od_id_point_dict[od_id] = point

    point_adj_dict = {}  # 以 point 对象为标识的邻接表
    used_point_list = set()
    for od_id in out_adj_table:
        point = od_id_point_dict[od_id]
        if point not in point_adj_dict:
            used_point_list.add(point)
            point_adj_dict[point] = []
        for to_id in out_adj_table[od_id]:
            used_point_list.add(od_id_point_dict[to_id])
            point_adj_dict[point].append(od_id_point_dict[to_id])
    point_list = list(used_point_list)

    g = Graph()
    for point in point_list:
        g.addVertex(point)
    for point in point_adj_dict:
        edge = []
        for to_point in point_adj_dict[point]:
            edge.append([to_point, 1])
        g.addDirectLine(point, edge)
    line_graph = g.getLineGraph()
    print('预测线图点个数', len(line_graph.nodes))
    print('预测线图边个数', len(line_graph.edges))

    force_nodes = []
    for node in line_graph.nodes:
        force_nodes.append({'name': f'{node[0]}_{node[1]}'})
    force_edges = []
    for edge in line_graph.edges:
        p1, p2 = edge[0], edge[1]
        force_edges.append({'source': f'{p1[0]}_{p1[1]}', 'target': f'{p2[0]}_{p2[1]}'})

    return force_nodes, force_edges, out_adj_table, line_graph, od_flow_dict, od_idx_dict


def get_line_graph(region, day, hour):
    with_space_dist = False
    # 计算线图，返回适用于 d3 的结构和邻接表 ===========================
    # used_od_cells = set([1, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 21, 23, 25, 26, 27, 28, 29, 31, 32, 33, 35, 36, 37, 38, 39, 41, 42, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 89, 91, 93, 94, 95, 96, 99])
    # tmp = {}
    # for start in out_adj_table:
    #     if start in used_od_cells:
    #         t = out_adj_table[start]
    #         t = list(set(t).intersection(used_od_cells))
    #         tmp[start] = t
    # out_adj_table = tmp
    cid_center_coord_dict = get_cell_id_center_coord_dict(region)
    # selected_cluster_ids = list(cid_center_coord_dict.keys())
    # selected_cluster_ids = set(cid_center_coord_dict.keys())
    # selected_cluster_ids = list(selected_cluster_ids.intersection(used_od_cells))
    # force_nodes, force_edges, filtered_adj_dict, lg = get_line_graph_by_selected_cluster(selected_cluster_ids, selected_cluster_ids, out_adj_table, exp_od_pair_set)
    od_mat = get_od_mat_by_time(day, hour)
    force_nodes, force_edges, out_adj_table, lg, od_flow_dict, od_idx_dict = build_line_graph_by_od_matrix(od_mat)

    history_od_mat = [
        # get_od_mat_by_time(day, hour - 1),
        # get_od_mat_by_time(day, hour - 2),
        # get_od_mat_by_time(day - 1, hour),
        # get_od_mat_by_time(day-7, hour),
    ]

    time_slice = (day - 1) * 24 * 4 + hour * 4 - (7 * 24 * 4)
    for i in range(768-1):  # 用前200个时间片的流量作为一个OD对的特征
        time_slice -= 1
        history_od_mat.append(get_od_mat_by_time_slice(time_slice))

    for o_id in out_adj_table:
        for d_id in out_adj_table[o_id]:
            for i in range(len(history_od_mat)):
                cur_od_mat = history_od_mat[i]
                od_flow_dict[(o_id, d_id)].insert(0, float(cur_od_mat[od_idx_dict[o_id * 100 + d_id]]))
                # od_flow_dict[(o_id, d_id)].insert(0, float(cur_od_mat[od_idx_dict[d_id * 100 + o_id]]))
    # print('od_flow_dict=======>', od_flow_dict)
            # od_flow_dict[(o_id, d_id)]

    # print('边 ', lg.edges())
    # print('点 ', lg.nodes())

    #  计算簇中心坐标 ========================================
    # tmp = {}
    # for key in cluster_point_dict:
    #     if int(key) in used_od_cells:
    #         tmp[int(key)] = cluster_point_dict[key]
    # cluster_point_dict = tmp

    # total_od_points = cache.get('total_od_points')
    # total_od_points = od_pair_process.get_od_points_filter_by_day_and_hour(month, start_day, end_day, 0, 24)['od_points']
    # cid_center_coord_dict = get_cluster_center_coord(total_od_points, cluster_point_dict, selected_cluster_ids)

    # +++++++++++++++ 轨迹获取和特征 ++++++++++++++
    # node_label_dict = None
    # if os.path.exists(f'./read_trjs_{start_day}_{end_day}_{start_hour}_{end_hour}.pkl'):
    #     with open(f'./read_trjs_{start_day}_{end_day}_{start_hour}_{end_hour}.pkl', 'rb') as f:
    #         obj = pickle.loads(f.read())
    #         trj_idxs, node_names_trjId_dict = obj['trj_idxs'], obj['node_names_trjId_dict']
    # else:
    #     trj_idxs, node_names_trjId_dict = get_trj_ids_by_force_node(force_nodes, cluster_point_dict, total_od_points, region)
    #     with open(f'./read_trjs_{start_day}_{end_day}_{start_hour}_{end_hour}.pkl', 'wb') as f:
    #         picklestring = pickle.dumps({
    #             'trj_idxs': trj_idxs,
    #             'node_names_trjId_dict': node_names_trjId_dict
    #         })
    #         f.write(picklestring)
    #         f.close()
    # print('get_trj_ids_by_force_node')

    if os.path.isfile(args.best_model):
        print("=> loading best_model '{}'".format(args.best_model))
        best_model = torch.load(args.best_model)

    # print('trj len', len(trj_idxs))
    # print('node name len', len(node_names_trjId_dict.keys()))
    # node_names_trjFeats_dict = {}   # 节点名 -> 包含的轨迹特征数组的 map
    # trjId_node_name_dict = {}   # 轨迹ID -> 所在的节点名的 map
    # node_names_trj_dict = {}    # 节点名 -> gps 轨迹数组的 map
    # for node_name in node_names_trjId_dict:
    #     node_trj_idxs = node_names_trjId_dict[node_name]
    #     for trj_id in node_trj_idxs:
    #         trjId_node_name_dict[trj_id] = node_name

    # trj_idxs = list(trjId_node_name_dict.keys())  # 所有轨迹id, trjId 的形式为 {天}_{当天的轨迹id}，这是由于每新的一天，轨迹id都从0开始算
    # gps_trips = get_trips_by_ids(trj_idxs, month, start_day, end_day)

    # print('draw_cluster_in_trj_view======================')
    # draw_cluster_in_trj_view([1 for i in range(len(gps_trips))], gps_trips)
    # trj_feats = run_model2(args, gps_trips, best_model, trj_region)    # 特征数组，顺序与 trj_idxs 对应
    # print(f'轨迹id数= {len(trj_idxs)}, 轨迹数 = {len(gps_trips)}, 特征数 = {len(trj_feats)}')

    # for i in range(len(trj_idxs)):
    #     id = trj_idxs[i]
    #     feat = trj_feats[i]
    #     trip = gps_trips[i]
    #     node_name = trjId_node_name_dict[id]
    #     if node_name not in node_names_trjFeats_dict:
    #         node_names_trjFeats_dict[node_name] = []
    #         node_names_trj_dict[node_name] = []
    #     node_names_trjFeats_dict[node_name].append(feat)    # 得到每个节点对应的其包含的特征们
    #     node_names_trj_dict[node_name].append(trip)

    # total_num = 0
    # for name in node_names_trjFeats_dict:
    #     total_num += len(node_names_trjFeats_dict[name])
    #     # print(f"{name} 包含 {len(node_names_trjFeats_dict[name])} 条轨迹")
    # avg_num = total_num // len(node_names_trjFeats_dict.keys())

    # ============== GCC 社区发现代码 ===============
    adj_mat = get_adj_matrix(lg)  # 根据线图得到 csc稀疏矩阵类型的邻接矩阵
    # features, related_node_names = get_feature_list(lg, node_names_trjFeats_dict, avg_num)  # 根据线图节点顺序，整理一个节点向量数组，以及对应顺序的node name
    features = []
    node_names = []
    for node in lg.nodes:
        name = lg.nodes[node]['name']
        # print('name', name)
        src, tgt = name.split('-')
        src, tgt = int(src), int(tgt)
        key = (src, tgt)
        if key in od_flow_dict:
            features.append(np.asarray(od_flow_dict[key]))
            # features.append(np.random.random([768]))
            node_names.append(key)
    features = np.array(features)
    print('feature  =========> ', features)


    print(f'线图节点个数：{len(lg.nodes())}, 向量个数：{len(features)}')
    print('向量长度', len(features[0]))

    is_none_graph_baseline = False
    is_none_feat_baseline = False

    # if is_none_feat_baseline is True:
    #     shape = features[0].shape
    #     # print(features[0])
    #     features = []
    #     for i in range(len(related_node_names)):
    #         features.append(np.random.random(shape))
    #         # features.append(np.zeros(shape))
    #     features = np.array(features)
    #     # print(features[0])

    ######## 仅在做实验时需要这个 for 循环，否则不需要循环，执行一次即可\
    # tsne_points = []
    cluster_point_dict = {}
    for cluster_num in [5, 10, 20, 30, 40, 50]:
        if is_none_graph_baseline:
            pass
            # labels_dict, trj_labels = get_cluster_by_trj_feature(cluster_num, torch.from_numpy(features))
            # # print('labels_dict==============t', labels_dict)
            # # tsne_points = utils.DoTSNE_show(features, 2, trj_labels)
            # # print('tsne_points', len(tsne_points))
            # # print('labels_dict', labels_dict)
            # node_name_cluster_dict = {}
            # for i in labels_dict:
            #     label = labels_dict[i]
            #     if label not in cluster_point_dict:
            #         cluster_point_dict[label] = []
            #     # 在线图中度为 0 的散点，视为噪声，从社区中排除
            #     # if get_degree_by_node_name(lg, related_node_names[i]) > 0:
            #     cluster_point_dict[label].append(node_names[int(i)])
            #     node_name_cluster_dict[node_names[int(i)]] = label
            # print('实际有效社区个数: ', get_ok_cluster_num(cluster_point_dict))
            # exp4_log.append(f'实际有效社区个数: {get_ok_cluster_num(cluster_point_dict)}')
        else:
            trj_labels = run(adj_mat, features, cluster_num)  # 得到社区划分结果，索引对应 features 的索引顺序，值是社区 id
            trj_labels = trj_labels.numpy().tolist()
            node_name_cluster_dict = {}
            for i in range(len(trj_labels)):
                label = trj_labels[i]
                if label not in cluster_point_dict:
                    cluster_point_dict[label] = []
                # 在线图中度为 0 的散点，视为噪声，从社区中排除
                # if get_degree_by_node_name(lg, related_node_names[i]) > 0:
                cluster_point_dict[label].append(node_names[i])
                node_name_cluster_dict[node_names[i]] = label
            print('实际有效社区个数: ', get_ok_cluster_num(cluster_point_dict))
            exp4_log.append(f'实际有效社区个数: {get_ok_cluster_num(cluster_point_dict)}')
        print(f'=========> feat len={len(features)}  nodename len={len(node_names)}  label len={len(trj_labels)}')
        print(list(trj_labels))
        # dag_force_nodes, dag_force_edges = get_dag_from_community(cluster_point_dict, force_nodes)

        to_draw_od_dict = {}
        for label in cluster_point_dict:
            to_draw_od_dict[label] = []
            for node_name in cluster_point_dict[label]:
                to_draw_od_dict[label].append([node_name[0], node_name[1]])
                # to_draw_trips_dict[label].extend(node_names_trj_dict[node_name])
        print('to_draw_trips_dict', to_draw_od_dict)
        data_dict = draw_cluster_in_trj_view_new_exp4(to_draw_od_dict, cluster_num, region)
        with pd.ExcelWriter(f'./cluster_res/exp4_excel/{start_day}-{end_day}-{start_hour}-{end_hour}-od_{cluster_num}_cluster_data.xlsx') as writer:
            for cluster_id in data_dict:
                data_frame = data_dict[cluster_id]
                data_frame = pd.DataFrame(data_frame)
                data_frame.to_excel(writer, sheet_name=f'社区{cluster_id}', index=False)
        # tsne_points = utils.DoTSNE(features, 2, cluster_point_dict)
        print(len(lg.nodes))


        # print(f'====> 社区个数：{cluster_num}, Q = {Q(lg, node_name_cluster_dict)}')
        print(f'====> 社区个数：{cluster_num}, CON = {avg_CON(lg, cluster_point_dict, node_name_cluster_dict)}')
        # print(f'====> 社区个数：{cluster_num}, TPR = {avg_TPR(lg, cluster_point_dict)}')

    if is_none_graph_baseline:
        file_name = f'{exp4_log_name}_none_graph_baseline_Q.txt'
    elif is_none_feat_baseline:
        file_name = f'{exp4_log_name}_none_feat_baseline_Q.txt'
    else:
        file_name = f'{exp4_log_name}_our_Q.txt'
    f = open(file_name, 'w')
    for log in exp4_log:
        f.write(log + '\n')
    f.close()

    return {
        'force_nodes': force_nodes,
        'force_edges': force_edges,
        # 'filtered_adj_dict': filtered_adj_dict,
        'cid_center_coord_dict': cid_center_coord_dict,
        'community_group': cluster_point_dict,
        # 'tsne_points': tsne_points,
        'trj_labels': trj_labels,  # 每个节点（OD对）的社区label，与 tsne_points 顺序对应
        # 'related_node_names': related_node_names,
        # 'tmp_trj_idxs': related_node_names,  # 与 tsne_points 顺序对应,
        'node_name_cluster_dict': node_name_cluster_dict
        # 'tid_trip_dict': tid_trip_dict,
    }



if __name__ == '__main__':
    def print_time():  # 无限循环
        print('-------------------->>>>  start')
        while True: # 获取当前的时间
            current_time = time.ctime(time.time()) # 输出线程的名字和时间
            print('keep live', current_time)  # 休眠10分钟，即600秒 time.sleep(600)
            time.sleep(600)

    thread = threading.Thread(target=print_time)
    thread.start()

    od_region = get_region()
    # cell_id_center_coord_dict = get_cell_id_center_coord_dict(od_region)
    # for key in cell_id_center_coord_dict:
    #     print(key, cell_id_center_coord_dict[key])
    with open("/home/zhengxuan.lin/project/deepcluster/data/region.pkl", 'rb') as file:
        trj_region = pickle.loads(file.read())
    # makeVocab(trj_region, h5_files)
    # total_od_pairs = get_od_filter_by_day_and_hour(month, start_day, end_day, start_hour, end_hour, od_region)
    # print(total_od_pairs[0:3])
    # od_pairs, od_cell_set, od_pair_set, hot_od_gps_set = get_od_hot_cell(total_od_pairs, od_region, 1000, 0)
    # res = get_grid_split(od_region)
    # get_line_graph(od_region, trj_region, month, start_day, end_day, start_hour, end_hour, res['out_adj_table'], res['cluster_point_dict'])

    day = 14
    hour = 19
    od_mat = get_od_mat_by_time(day, hour)
    get_line_graph(od_region, day, hour)
    # data = np.load('./history_od_data.npz', allow_pickle=True)['data']
    # # print(data)
    # # print(data.shape)
    # time_slice = (day - 1) * 24 * 4 + hour * 4 - (7 * 24 * 4)
    # data = get_od_mat_by_time_slice(time_slice)
    # print(data)
    # print(data.shape)

