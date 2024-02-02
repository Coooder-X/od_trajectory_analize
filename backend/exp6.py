import os
import pickle
from datetime import datetime
import time
import threading

import numpy as np
import pandas as pd
import torch

import utils
from graph_process.Point import Point
from cal_od import exp_od_pair_set, get_od_filter_by_day_and_hour, get_od_hot_cell, encode_od_point
from data_process import od_pair_process
from data_process.OD_area_graph import build_od_graph, fuse_fake_edge_into_linegraph, \
    get_line_graph_by_selected_cluster
from data_process.SpatialRegionTools import get_cell_id_center_coord_dict, makeVocab, inregionS
from data_process.od_pair_process import get_trips_by_ids, get_trj_ids_by_force_node, \
    get_odpair_space_similarity
from data_process.spatial_grid_utils import get_region, get_od_points_filter_by_region, divide_od_into_grid
from gcc.graph_convolutional_clustering.gcc.run import run, draw_cluster_in_trj_view, draw_cluster_in_trj_view_new
from graph_process.Graph import get_degree_by_node_name, get_feature_list, get_adj_matrix, Graph
from t2vec import args
from t2vec_graph import run_model2, get_cluster_by_trj_feature
import networkx as nx

exp3_log_name = 'exp6_log'
exp3_log = []

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
    exp3_log.append(f'cluster_num {len(set(list(node_name_cluster_dict.values())))} Q = {res}')
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
        # print(f'u = {u}, v = {v}')
        u_name, v_name = f'{u[0]}_{u[1]}', f'{v[0]}_{v[1]}'
        if (u_name in node_name_cluster_dict and node_name_cluster_dict[u_name] == cluster_id and (v_name not in node_name_cluster_dict or (v_name in node_name_cluster_dict and node_name_cluster_dict[v_name] != cluster_id))) \
            or \
            (v_name in node_name_cluster_dict and node_name_cluster_dict[v_name] == cluster_id and (u_name not in node_name_cluster_dict or (u_name in node_name_cluster_dict and node_name_cluster_dict[u_name] != cluster_id))):
        # if u_name in node_name_cluster_dict and v_name in node_name_cluster_dict and ((node_name_cluster_dict[u_name] == cluster_id and node_name_cluster_dict[v_name] != cluster_id) or \
        #     (node_name_cluster_dict[u_name] != cluster_id and node_name_cluster_dict[v_name] == cluster_id)):
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
        name = f'{src}_{tgt}'
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
        print("==========> cur_con =", cur_con)
        if cur_con == -1:
            continue
        ok_cluster_num += 1
        avg += cur_con
        print(f'cluster: {cluster_id} cur_con = {cur_con}')

    avg /= ok_cluster_num
    exp3_log.append(f'cluster_num {len(cluster_point_dict.keys())} avg Con = {avg}')
    return avg


def get_ok_cluster_num(cluster_point_dict):
    ok_cluster_num = 0
    for cluster_id in cluster_point_dict:
        if len(cluster_point_dict[cluster_id]) > 5:
            ok_cluster_num += 1
    return ok_cluster_num


def get_origin_graph_by_selected_cluster(selected_cluster_ids_in_brush, selected_cluster_ids, out_adj_dict, exp_od_pair_set):
    """
    :param selected_cluster_ids_in_brush: 一个数组，存储地图中选取框框内的簇的id
    :param selected_cluster_ids: 一个数组，存储地图中已选的所有簇的id
    :param out_adj_dict: 当天数据中所有簇的全量的邻接表，out_adj_dict[x] 存储簇 id 为 x 的簇，会到达的簇的 id 数组
    :exp_od_pair_set: 一个set，包含一些OD对，只有在这个集合中的OD对才可以被用于建图
    :return g: 原图的 networkx 对象
    :return filtered_adj_dict: 根据已选簇，从全量簇的邻接表中过滤出的已选簇的邻接表
    """
    # selected_cluster_ids, out_adj_dict = cids, adj
    # selected_cluster_ids = list(set(selected_cluster_ids))
    # print(selected_cluster_ids)
    # 过滤出邻接表中有用的部分
    filtered_adj_dict = {}  # 用已选簇id过滤后的邻接表，索引是簇id
    for cid in selected_cluster_ids:
        if cid not in filtered_adj_dict:
            filtered_adj_dict[cid] = []
        # 如果 to_cid 是 cid 的邻接点，则应该加入【过滤邻接表】中
        for to_cid in selected_cluster_ids:
            if to_cid == cid:
                continue
            # 如果起终点都不在地图选取框框内的，就过滤掉
            if cid not in selected_cluster_ids_in_brush and to_cid not in selected_cluster_ids_in_brush:
                continue
            # if cid in out_adj_dict and to_cid in out_adj_dict[cid] and \
            #         (cid, to_cid) in exp_od_pair_set:
            if cid in out_adj_dict and to_cid in out_adj_dict[cid]:
                filtered_adj_dict[cid].append(to_cid)

    cluster_list = []   # 存储所有 Point 类型的 簇，作为 graph 的节点集
    cid_point_dict = {}  # 簇id 到 Point 类型的簇 的映射

    for cid in selected_cluster_ids:
        point = Point(name=cid, nodeId=cid, infoObj={}, feature={})
        cluster_list.append(point)
        cid_point_dict[cid] = point

    adj_point_dict = {}  # 根据 filtered_adj_dict 得出的等价的邻接表，索引是 Point 类型的簇
    for cid in filtered_adj_dict:
        point = cid_point_dict[cid]
        if point not in adj_point_dict:
            adj_point_dict[point] = []
        for to_cid in filtered_adj_dict[cid]:
            adj_point_dict[point].append(cid_point_dict[to_cid])

    g = Graph()
    for cluster in cluster_list:
        g.addVertex(cluster)
    #   边权值可以后续改成簇之间的 od 对数量，暂时默认为 1
    for point in adj_point_dict:
        edge = []
        for to_point in adj_point_dict[point]:
            edge.append([to_point, 1])
        g.addDirectLine(point, edge)
    return g, filtered_adj_dict
    # line_graph = g.getLineGraph()
    # g.drawGraph()
    # g.drawLineGraph()
    # print(line_graph.nodes)
    # print(line_graph.edges)
    # print('点数据', line_graph.nodes.data())
    # print('点个数', len(line_graph.nodes))
    # print('边个数', len(line_graph.edges))

    # force_nodes = []
    # for node in line_graph.nodes:
    #     force_nodes.append({ 'name': f'{node[0]}_{node[1]}' })
    # force_edges = []
    # for edge in line_graph.edges:
    #     p1, p2 = edge[0], edge[1]
    #     force_edges.append({ 'source': f'{p1[0]}_{p1[1]}', 'target': f'{p2[0]}_{p2[1]}' })
    # return force_nodes, force_edges, filtered_adj_dict, line_graph


month = 5
# start_day, end_day = 11, 12
start_day, end_day = 12, 15
# start_hour, end_hour = 18, 20
start_hour, end_hour = 8, 10


def get_edge_cluster_by_point_cluster(g, lg, related_node_names, cluster_point_dict, node_name_cluster_dict):
    """
    已知原图上进行社区发现的结果，根据原图中的节点社区，将社区中的边抽出，对应到线图中的节点集合，得到线图中的节点划分。
    """
    # print(f'=======> 原图的边：g.edges = {g.edges}')
    # print(f'=======> 线图的点：lg.nodes = {lg.nodes}')
    lg_point_set = set(list(lg.nodes))  # 线图的节点集合
    lg_cluster_point_dict = {}  # 线图的 cluster_point_dict
    lg_node_name_cluster_dict = {}
    g_cluster_edge_dict = {}    # 原图的节点社区中，抽出的社区内的边
    g_edge_set = set(list(g.edges)) # 原图的边集合
    for cluster_id in cluster_point_dict:
        g_cluster_edge_dict[cluster_id] = []
        point_set = cluster_point_dict[cluster_id]
        # 遍历某个原图社区中的任意两个点，若中间存在一条边，则加入原图边社区
        for i in point_set:
            for j in point_set:
                edge1 = (i, j, 0)
                edge2 = (j, i, 0)
                if edge1 in g_edge_set:
                    g_cluster_edge_dict[cluster_id].append(edge1)
                if edge2 in g_edge_set:
                    g_cluster_edge_dict[cluster_id].append(edge2)
        if len(g_cluster_edge_dict[cluster_id]) == 0:
            del g_cluster_edge_dict[cluster_id]
        else:
            # print(f'========> g_cluster_edge_dict = {g_cluster_edge_dict[cluster_id]}')
            # 得到原图边社区后，遍历当前社区内的边，得到对应的线图中的节点
            for edge in g_cluster_edge_dict[cluster_id]:
                if edge in lg_point_set:
                    if cluster_id not in lg_cluster_point_dict:
                        lg_cluster_point_dict[cluster_id] = []
                    lg_cluster_point_dict[cluster_id].append(edge)
                    node_name = f'{edge[0]}_{edge[1]}'
                    lg_node_name_cluster_dict[node_name] = cluster_id
            print(f'=============> lg_cluster_point_dict = {lg_cluster_point_dict[cluster_id]}')
    print(f'===========> lg_node_name_cluster_dict.keys() = {lg_node_name_cluster_dict.keys()}')
    return lg_cluster_point_dict, lg_node_name_cluster_dict


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


def get_line_graph(region, trj_region, month, start_day, end_day, start_hour, end_hour, out_adj_table, cluster_point_dict, exp_od_pair_set):
    with_space_dist = False
    # 计算线图，返回适用于 d3 的结构和邻接表 ===========================
    used_od_cells = set([1, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 21, 23, 25, 26, 27, 28, 29, 31, 32, 33, 35, 36, 37, 38, 39, 41, 42, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 89, 91, 93, 94, 95, 96, 99])
    tmp = {}
    for start in out_adj_table:
        if start in used_od_cells:
            t = out_adj_table[start]
            t = list(set(t).intersection(used_od_cells))
            tmp[start] = t
    out_adj_table = tmp
    cid_center_coord_dict = get_cell_id_center_coord_dict(region)
    # selected_cluster_ids = list(cid_center_coord_dict.keys())
    selected_cluster_ids = set(cid_center_coord_dict.keys())
    selected_cluster_ids = list(selected_cluster_ids.intersection(used_od_cells))
    g, filtered_adj_dict = get_origin_graph_by_selected_cluster(selected_cluster_ids, selected_cluster_ids, out_adj_table, exp_od_pair_set)
    force_nodes, force_edges, line_graph_filtered_adj_dict, lg = get_line_graph_by_selected_cluster(selected_cluster_ids, selected_cluster_ids, out_adj_table, exp_od_pair_set)
    g = g.G

    # print('边 ', lg.edges())
    # print('点 ', lg.nodes())

    #  计算簇中心坐标 ========================================
    tmp = {}
    for key in cluster_point_dict:
        if int(key) in used_od_cells:
            tmp[int(key)] = cluster_point_dict[key]
    cluster_point_dict = tmp

    total_od_points = od_pair_process.get_od_points_filter_by_day_and_hour(month, start_day, end_day, 0, 24)['od_points']
    # cid_center_coord_dict = get_cluster_center_coord(total_od_points, cluster_point_dict, selected_cluster_ids)

    # # +++++++++++++++ 轨迹获取和特征 ++++++++++++++
    # # node_label_dict = None
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
    #
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
    adj_mat = get_adj_matrix(g)  # 根据线图得到 csc稀疏矩阵类型的邻接矩阵
    # features, related_node_names = get_feature_list(lg, node_names_trjFeats_dict, avg_num)  # 根据线图节点顺序，整理一个节点向量数组，以及对应顺序的node name

    print(f'原图节点个数：{len(g.nodes())}')
    # print('向量长度', len(features[0]))

    is_none_graph_baseline = False
    is_none_feat_baseline = True

    print(f'===>> g.nodes = {g.nodes}')
    related_node_names = list(g.nodes())
    print(list(g.nodes()))
    if is_none_feat_baseline is True:
        shape = [768]  # features[0].shape
        # print(features[0])
        features = []
        # related_node_names = []
        for node in g.nodes():
            features.append(np.random.random(shape))
            # print(f'---> g.nodes[i] = {g.nodes[node]}')
            # related_node_names.append(g.nodes[node])
            # features.append(np.zeros(shape))
        features = np.array(features)
        # print(features[0])

    ######## 仅在做实验时需要这个 for 循环，否则不需要循环，执行一次即可\
    tsne_points = []
    cluster_point_dict = {}
    for cluster_num in [10, 20, 30, 40, 50]:
        if is_none_graph_baseline:
            pass
            # labels_dict, trj_labels = get_cluster_by_trj_feature(cluster_num, torch.from_numpy(features))
            # # print('labels_dict==============t', labels_dict)
            # tsne_points = utils.DoTSNE_show(features, 2, trj_labels)
            # print('tsne_points', len(tsne_points))
            # # print('labels_dict', labels_dict)
            # node_name_cluster_dict = {}
            # for i in labels_dict:
            #     label = labels_dict[i]
            #     if label not in cluster_point_dict:
            #         cluster_point_dict[label] = []
            #     # 在线图中度为 0 的散点，视为噪声，从社区中排除
            #     # if get_degree_by_node_name(lg, related_node_names[i]) > 0:
            #     cluster_point_dict[label].append(related_node_names[int(i)])
            #     node_name_cluster_dict[related_node_names[int(i)]] = label
            # print('实际有效社区个数: ', get_ok_cluster_num(cluster_point_dict))
            # exp3_log.append(f'实际有效社区个数: {get_ok_cluster_num(cluster_point_dict)}')
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
                cluster_point_dict[label].append(related_node_names[i])
                node_name_cluster_dict[related_node_names[i]] = label

            print('原图有效社区个数: ', get_ok_cluster_num(cluster_point_dict))
            exp3_log.append(f'原图有效社区个数: {get_ok_cluster_num(cluster_point_dict)}')
            cluster_point_dict, node_name_cluster_dict = get_edge_cluster_by_point_cluster(g, lg, related_node_names, cluster_point_dict, node_name_cluster_dict)

            print('线图有效社区个数: ', get_ok_cluster_num(cluster_point_dict))
            exp3_log.append(f'线图有效社区个数: {get_ok_cluster_num(cluster_point_dict)}')
        print(f'=========> feat len={len(features)}  nodename len={len(related_node_names)}  label len={len(trj_labels)}')
        print(list(trj_labels))
        # dag_force_nodes, dag_force_edges = get_dag_from_community(cluster_point_dict, force_nodes)

        # to_draw_trips_dict = {}
        # for label in cluster_point_dict:
        #     to_draw_trips_dict[label] = []
        #     for node_name in cluster_point_dict[label]:
        #         to_draw_trips_dict[label].extend(node_names_trj_dict[node_name])
        # print('to_draw_trips_dict', to_draw_trips_dict)
        # data_dict, od_dict = draw_cluster_in_trj_view_new(to_draw_trips_dict, cluster_num, region)
        # with pd.ExcelWriter(f'./cluster_res/excel/{start_day}-{end_day}-{start_hour}-{end_hour}-od_{cluster_num}_cluster_data.xlsx') as writer:
        #     for cluster_id in data_dict:
        #         data_frame = data_dict[cluster_id]
        #         data_frame = pd.DataFrame(data_frame)
        #         data_frame.to_excel(writer, sheet_name=f'社区{cluster_id}', index=False)
        #         od_data_frame = od_dict[cluster_id]
        #         od_data_frame = pd.DataFrame(od_data_frame)
        #         od_data_frame.to_excel(writer, sheet_name=f'社区{cluster_id}_od点', index=False)
        # tsne_points = utils.DoTSNE(features, 2, cluster_point_dict)
        print(len(g.nodes))


        # print(f'====> 社区个数：{cluster_num}, Q = {Q(lg, node_name_cluster_dict)}')
        print(f'====> 社区个数：{cluster_num}, CON = {avg_CON(lg, cluster_point_dict, node_name_cluster_dict)}')
        # print(f'====> 社区个数：{cluster_num}, TPR = {avg_TPR(lg, cluster_point_dict)}')

    if is_none_graph_baseline:
        file_name = f'{exp3_log_name}_none_graph_baseline_Q.txt'
    elif is_none_feat_baseline:
        file_name = f'{exp3_log_name}_none_feat_baseline_Q.txt'
    else:
        file_name = f'{exp3_log_name}_our_Q.txt'
    f = open(file_name, 'w')
    for log in exp3_log:
        f.write(log + '\n')
    f.close()

    return {
        # 'force_nodes': force_nodes,
        # 'force_edges': force_edges,
        'filtered_adj_dict': filtered_adj_dict,
        'cid_center_coord_dict': cid_center_coord_dict,
        'community_group': cluster_point_dict,
        'tsne_points': tsne_points,
        'trj_labels': trj_labels,  # 每个节点（OD对）的社区label，与 tsne_points 顺序对应
        'related_node_names': related_node_names,
        'tmp_trj_idxs': related_node_names,  # 与 tsne_points 顺序对应,
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
    with open("/home/zhengxuan.lin/project/od_trajectory_analize/backend/data/region.pkl", 'rb') as file:
        trj_region = pickle.loads(file.read())
    # makeVocab(trj_region, h5_files)
    total_od_pairs = get_od_filter_by_day_and_hour(month, start_day, end_day, start_hour, end_hour, od_region)
    # print(total_od_pairs[0:3])
    od_pairs, od_cell_set, od_pair_set, hot_od_gps_set = get_od_hot_cell(total_od_pairs, od_region, 1000, 0)
    res = get_grid_split(od_region, od_pair_set, hot_od_gps_set)
    get_line_graph(od_region, trj_region, month, start_day, end_day, start_hour, end_hour, res['out_adj_table'], res['cluster_point_dict'], od_pair_set)


