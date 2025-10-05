import os
import pickle
from datetime import datetime
import time
import threading

# from cdlib import algorithms
import igraph as ig
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
from graph_process.Graph import get_degree_by_node_name, get_feature_list, get_adj_matrix, Graph, networkx2igraph, igraph2networkx
from t2vec import args
from t2vec_graph import run_model2, get_cluster_by_trj_feature
import networkx as nx
from MAGI.magi import run_magi
from HLC.hlc_utils import run_cdlib_hlc, detect_line_graph_nodes, map_edge_communities_to_nodes, merge_clusters_to_target
try:
    from cdlib import algorithms
    _cdlib_available = True
except Exception:
    _cdlib_available = False

exp5_log_name = 'exp5_log'
exp5_log = []

args.cuda = False
consider_edge_weight = True
use_line_graph = False
use_igraph = False
tradition_method = 'CNM'  # 'CNM' 'louvain'
use_magi = False  # 是否使用MAGI方法（2024年新方法）
# 是否使用CDlib的链接社区检测（2010年的方法，直接对原图进行边聚类。运行此方法时 use_line_graph 要为 False）
use_cdlib_link_community = True

# CDlib(0.4.0) 后处理参数：
need_merge_hlc_cluster = True   # 是否启用合并：将碎片化小社区合并到目标簇数；False 则直接输出 HLC 原始结果（仅做必要的格式映射）
# 是否过滤：边社区的最小边数（在映射之前过滤），若开启，则过滤掉社区内边数小于阈值的社区，保证社区内边的条数不小于阈值
need_filter_small_edge = False
cdlib_min_edges = 2            # 过滤：边社区的最小边数（在映射之前过滤）
cdlib_target_k = 10            # 目标社区数（合并后尽量逼近）
cdlib_merge_max_iter = 300     # 最大合并迭代次数
cdlib_merge_min_overlap = 0.0  # 合并阈值：最小Jaccard重叠；0表示总能合并（防止卡死）

month = 5
start_day, end_day = 12, 14
start_hour, end_hour = 8, 10
# start_day, end_day = 11, 12
# start_hour, end_hour = 18, 20


def CON(G, cluster_id, node_name_cluster_dict):
    start = datetime.now()
    m = len(G.edges())
    fz = 0
    for edge in G.edges():
        u, v = edge[0], edge[1]
        u_name, v_name = u, v
        # u_name, v_name = f'{u[0]}_{u[1]}', f'{v[0]}_{v[1]}'
        u_c = node_name_cluster_dict.get(u_name, None)
        v_c = node_name_cluster_dict.get(v_name, None)
        if (u_c == cluster_id and v_c != cluster_id) or \
                (u_c != cluster_id and v_c == cluster_id):
            fz += 1
    vol_C = vol(G, cluster_id, node_name_cluster_dict)
    # print(f'vol_C={vol_C}({cluster_id})')
    fm = fz + vol_C
    # fm = min(vol_C, m - vol_C)
    if fm == 0 or fz == 0:
        return -1
    end = datetime.now()
    # print('用时', end - start)
    # print(f'分子={fz}， 分母={fm}')
    # res = fz / (fz + vol_C + 0.01)
    # print(f'CON=({res})')
    res = fz / fm
    return res


def vol(G, cluster_id, node_name_cluster_dict):
    res = 0
    # print(f'vol ==== G.nodes() = {G.nodes()}')
    for node in G.nodes():
        if node in node_name_cluster_dict and node_name_cluster_dict[node] == cluster_id:
            res += G.degree(node)
    return res


def avg_CON(G, cluster_point_dict, node_name_cluster_dict, use_igraph):
    if use_igraph is True:
        G = igraph2networkx(G, nx.MultiDiGraph)
    avg = 0.0
    ok_cluster_num = 0
    for cluster_id in cluster_point_dict:
        # if len(cluster_point_dict[cluster_id]) > 5:
        cur_con = CON(G, cluster_id, node_name_cluster_dict)
        if cur_con == -1:
            continue
        ok_cluster_num += 1
        avg += cur_con
        print(f'cluster: {cluster_id} cur_con = {cur_con}')

    if ok_cluster_num == 0:
        exp5_log.append(f'cluster_num {len(cluster_point_dict.keys())} avg Con：有效的社区个数为0，无法计算 CON')
        return '有效的社区个数为0，无法计算 CON'
    avg /= ok_cluster_num
    exp5_log.append(f'cluster_num {len(cluster_point_dict.keys())} avg Con = {avg}')
    return avg


def get_ok_cluster_num(cluster_point_dict):
    ok_cluster_num = 0
    for cluster_id in cluster_point_dict:
        if len(cluster_point_dict[cluster_id]) > 5:
            ok_cluster_num += 1
    return ok_cluster_num


# CDlib 链接社区检测，是基于原图，但对边进行聚类的方法。因此社区内有2个元素就是有效的社区。因为2个元素的含义是2条边，已经包含了2个节点
def get_ok_cluster_num_for_line_graph_cdlib(cluster_point_dict):
    ok_cluster_num = 0
    for cluster_id in cluster_point_dict:
        if len(cluster_point_dict[cluster_id]) >= 2:
            ok_cluster_num += 1
    return ok_cluster_num


def get_origin_graph_by_selected_cluster(selected_cluster_ids_in_brush, selected_cluster_ids, out_adj_dict,
                                         exp_od_pair_set):
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
    with open(f'./od_flow_dict.pkl', 'rb') as f:
        obj = pickle.loads(f.read())
        print('obj============', obj)
        od_flow_dict = obj['od_flow_dict']
        f.close()
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

    cluster_list = []  # 存储所有 Point 类型的 簇，作为 graph 的节点集
    cid_point_dict = {}  # 簇id 到 Point 类型的簇 的映射
    point_cid_dict = {}  # Point 类型的簇 到 簇id 的映射

    for cid in selected_cluster_ids:
        point = Point(name=cid, nodeId=cid, infoObj={}, feature={})
        cluster_list.append(point)
        cid_point_dict[cid] = point
        point_cid_dict[point] = cid

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
        u = point_cid_dict[point]
        for to_point in adj_point_dict[point]:
            v = point_cid_dict[to_point]
            if consider_edge_weight is True:
                if (u, v) in od_flow_dict:
                    edge.append([to_point, od_flow_dict[(u, v)]])
                else:
                    print(f'==========>>>> {(u, v)} 不在总的OD流中')
            else:
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


def get_line_graph(region, trj_region, month, start_day, end_day, start_hour, end_hour, out_adj_table,
                   cluster_point_dict, exp_od_pair_set):
    with_space_dist = False
    # 计算线图，返回适用于 d3 的结构和邻接表 ===========================
    used_od_cells = set(
        [1, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 21, 23, 25, 26, 27, 28, 29, 31, 32, 33, 35, 36, 37, 38, 39, 41, 42, 44,
         45, 46, 47, 48, 49, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73,
         74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 89, 91, 93, 94, 95, 96, 99])
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
    if use_line_graph is True:
        g, filtered_adj_dict = get_origin_graph_by_selected_cluster(selected_cluster_ids, selected_cluster_ids,
                                                                    out_adj_table, exp_od_pair_set)
        # 保留原始图（用于CDlib的链接社区检测）
        original_g_nx = g.G
        force_nodes, force_edges, line_graph_filtered_adj_dict, lg = get_line_graph_by_selected_cluster(
            selected_cluster_ids, selected_cluster_ids, out_adj_table, exp_od_pair_set)
        if use_igraph is True:
            g = g.G
            g_tmp = ig.Graph(directed=True)
            g_tmp = g_tmp.from_networkx(g)
            g = g_tmp.linegraph()
        else:
            g = lg
    else:
        g, filtered_adj_dict = get_origin_graph_by_selected_cluster(selected_cluster_ids, selected_cluster_ids,
                                                                    out_adj_table, exp_od_pair_set)
        if use_igraph is True:
            g = networkx2igraph(g.G)
        else:
            g = g.G

    # print('边 ', lg.edges())
    # print('点 ', lg.nodes())

    #  计算簇中心坐标 ========================================
    tmp = {}
    for key in cluster_point_dict:
        if int(key) in used_od_cells:
            tmp[int(key)] = cluster_point_dict[key]
    cluster_point_dict = tmp

    total_od_points = od_pair_process.get_od_points_filter_by_day_and_hour(month, start_day, end_day, 0, 24)[
        'od_points']

    # # +++++++++++++++ 轨迹获取和特征 ++++++++++++++
    if use_line_graph:
        trj_idxs, node_names_trjId_dict = get_trj_ids_by_force_node(force_nodes, cluster_point_dict, total_od_points, region)

        best_model = None
        print('os.path.isfile(args.best_model)', os.path.isfile(args.best_model))
        if os.path.isfile(args.best_model):
            print("=> loading best_model '{}'".format(args.best_model))
            if args.cuda:
                best_model = torch.load(args.best_model, weights_only=False)
            else:
                best_model = torch.load(args.best_model, map_location=torch.device('cpu'), weights_only=False)

        node_names_trjFeats_dict = {}   # 节点名 -> 包含的轨迹特征数组的 map
        trjId_node_name_dict = {}   # 轨迹ID -> 所在的节点名的 map
        node_names_trj_dict = {}    # 节点名 -> gps 轨迹数组的 map
        for node_name in node_names_trjId_dict:
            node_trj_idxs = node_names_trjId_dict[node_name]
            for trj_id in node_trj_idxs:
                trjId_node_name_dict[trj_id] = node_name

        trj_idxs = list(trjId_node_name_dict.keys())  # 所有轨迹id, trjId 的形式为 {天}_{当天的轨迹id}，这是由于每新的一天，轨迹id都从0开始算
        gps_trips = get_trips_by_ids(trj_idxs, month, start_day, end_day)
        #
        # print('draw_cluster_in_trj_view======================')
        # draw_cluster_in_trj_view([1 for i in range(len(gps_trips))], gps_trips)
        trj_feats = run_model2(args, gps_trips, best_model, trj_region)    # 特征数组，顺序与 trj_idxs 对应
        # print(f'轨迹id数= {len(trj_idxs)}, 轨迹数 = {len(gps_trips)}, 特征数 = {len(trj_feats)}')

        for i in range(len(trj_idxs)):
            id = trj_idxs[i]
            feat = trj_feats[i]
            trip = gps_trips[i]
            node_name = trjId_node_name_dict[id]
            if node_name not in node_names_trjFeats_dict:
                node_names_trjFeats_dict[node_name] = []
                node_names_trj_dict[node_name] = []
            node_names_trjFeats_dict[node_name].append(feat)    # 得到每个节点对应的其包含的特征们
            node_names_trj_dict[node_name].append(trip)

        total_num = 0
        for name in node_names_trjFeats_dict:
            total_num += len(node_names_trjFeats_dict[name])
            # print(f"{name} 包含 {len(node_names_trjFeats_dict[name])} 条轨迹")
        avg_num = total_num // len(node_names_trjFeats_dict.keys())

        # ============== GCC 社区发现代码 ===============
        adj_mat = get_adj_matrix(g)  # 根据线图得到 csc稀疏矩阵类型的邻接矩阵
        features, related_node_names = get_feature_list(lg, node_names_trjFeats_dict, avg_num)  # 根据线图节点顺序，整理一个节点向量数组，以及对应顺序的node name

    # print(f'原图节点个数：{len(g.nodes())}')
    # print('向量长度', len(features[0]))

    related_node_names = list(g.nodes())

    ######## 仅在做实验时需要这个 for 循环，否则不需要循环，执行一次即可\
    tsne_points = []
    cluster_point_dict = {}
    weight = 'edge_feature' if consider_edge_weight is True else None
    # for cluster_num in [10, 20, 30, 40, 50]:
    for cluster_num in [5, 10, 20, 30, 40, 50]:
        # 使用CDlib的链接社区（边社区）方法：在原图上做边聚类，再映射为线图上的节点社区
        if use_cdlib_link_community:
            cluster_point_dict = {}
            node_name_cluster_dict = {}
            if not _cdlib_available:
                print('CDlib 未安装，跳过链接社区检测（pip install cdlib）')
                return
            try:
                # 若未构建线图，则使用当前 g 的对应原始图
                # original_g_nx 在 use_line_graph=True 时已保留；否则 g 可能已是原图
                cdlib_input_graph = original_g_nx if 'original_g_nx' in locals() and original_g_nx is not None else g
                # 兼容不同版本CDlib：优先使用 full，不存在或签名不兼容则回退到基础版本/无参数
                # 由于 hierarchical_link_community 方法不支持指定簇数k，探测的社区个数和有效社区个数差距较大（前者过大、后者过小），
                # 因此，需要进行后处理，合并小的社区，直到社区个数达到 k（或目标范围）。
                ec = run_cdlib_hlc(cdlib_input_graph)
                # 映射/后处理：
                # 1) 边社区集合获取；当启用后处理时按 need_filter_small_edge 过滤；关闭后处理则直接使用原始边社区（不做过滤）。
                raw_edge_comms = list(ec.communities)
                edge_comms = (
                    [c for c in raw_edge_comms if len(c) >= cdlib_min_edges]
                    if need_filter_small_edge else
                    raw_edge_comms
                )
                # 将边社区映射为：
                # - 如果当前 g 是线图（节点是边tuple），则直接用边tuple作为社区节点
                # - 如果当前 g 是原图（节点是原节点id），则将边社区转换为其端点节点的并集
                is_line_graph_nodes, current_nodes = detect_line_graph_nodes(g)
                # 本实验中，如果输出的两个都是 false，则 hierarchical_link_community 方法的使用是符合预期的
                print(f'是否是基于线图运行: is_line_graph_nodes: {is_line_graph_nodes}, use_line_graph: {use_line_graph}')
                cluster_point_dict, node_name_cluster_dict = map_edge_communities_to_nodes(
                    edge_comms, g, is_line_graph_nodes, current_nodes
                )
                # 2) 合并：基于节点集合的Jaccard重叠，计算两个社区之间的 Jaccard 相似度，迭代合并最小簇到重叠度最高的簇
                # 以当前循环的 cluster_num 为目标簇数进行合并（保持与原逻辑一致）
                if need_merge_hlc_cluster and len(cluster_point_dict) > cluster_num:
                    cluster_point_dict, node_name_cluster_dict = merge_clusters_to_target(
                        cluster_point_dict,
                        target_k=cluster_num,
                        max_iter=cdlib_merge_max_iter,
                        min_overlap=cdlib_merge_min_overlap,
                    )
                # 使用得到的社区数
                final_cluster_num = len(cluster_point_dict.keys())
                print('CDlib 链接社区结果: ', cluster_point_dict)
                exp5_log.append(f'CDlib链接社区，设定k={cluster_num} 实际有效社区个数: {get_ok_cluster_num_for_line_graph_cdlib(cluster_point_dict)}')
            except Exception as e:
                print('CDlib 链接社区检测失败: ', e)
            # 评估并进入下一轮
            print(f'====> 社区个数：{final_cluster_num}, CON = {avg_CON(g, cluster_point_dict, node_name_cluster_dict, use_igraph)}')
            continue
        if tradition_method == 'louvain':
            # louvain --------------------------------------------------------------------
            communities = nx.algorithms.community.louvain_partitions(g, weight=weight, resolution=0.7, threshold=1e-03, seed=30)
            trj_labels = []
            for c in communities:
                trj_labels.append(c)
            print('trj==', trj_labels)
            communities = trj_labels[0]
            print('=====> 社区划分结果：', communities)
            cluster_num = len(communities)
            node_name_cluster_dict = {}
            cluster_point_dict = {}
            for (i, cluster) in enumerate(communities):
                cluster_point_dict[i] = list(cluster)
                for cluster_id in cluster:
                    node_name_cluster_dict[cluster_id] = i

            # em --------------------------------------------------------------------------
        # if tradition_method == 'em':
            # communities = algorithms.em(g, cluster_num)
            # communities = algorithms.async_fluid(g, cluster_num)
            # g_tmp = ig.Graph(directed=True)
            # g_tmp.add_vertices(list(g.nodes))
            # g_tmp.add_edges(list(g.edges))
            # g = g_tmp
            # communities = g.community_edge_betweenness(clusters=cluster_num, directed=True, weights=None)
            # print('=====> communities1=', communities)
            # trj_labels = communities
            # communities = list(communities.communities)
            # node_name_cluster_dict = {}
            # cluster_point_dict = {}
            # for (i, cluster) in enumerate(communities):
            #     cluster_point_dict[i] = list(cluster)
            #     for cluster_id in cluster:
            #         node_name_cluster_dict[cluster_id] = i

            # community_edge_betweenness (igraph)  ------------------------------------------------------
        # if tradition_method == 'community_edge_betweenness':
            # com = g.community_edge_betweenness(clusters=cluster_num, directed=True, weights=None)
            # # com = g.community_leading_eigenvector(clusters=cluster_num, arpack_options=None, weights=None)
            # # print('com is ==============>', com)
            # # com = ig.GraphBase.community_edge_betweenness(g, 3, True)
            # print('com ===========>', com.as_clustering())
            # print('com ===========>', com)
            # communities = com.as_clustering()
            # trj_labels = communities
            # # communities = list(communities.communities)
            # node_name_cluster_dict = {}
            # cluster_point_dict = {}
            # for (i, cluster) in enumerate(communities):
            #     cluster_point_dict[i] = list(cluster)
            #     for cluster_id in cluster:
            #         node_name_cluster_dict[cluster_id] = i

            # asyn_lpa_communities --------------------------------------------------------
        # if tradition_method == 'asyn_lpa_communities':
            # communities = networkx.algorithms.community.asyn_lpa_communities(g, weight=weight, seed=None)
            # print('=====> communities1=', communities)
            # trj_labels = []
            # for c in communities:
            #     trj_labels.append(c)
            # print('trj==', trj_labels)
            # communities = trj_labels
            # print('=====> communities2=', communities)
            # node_name_cluster_dict = {}
            # cluster_point_dict = {}
            # for (i, cluster) in enumerate(communities):
            #     cluster_point_dict[i] = list(cluster)
            #     for cluster_id in cluster:
            #         node_name_cluster_dict[cluster_id] = i

        # greedy_modularity_communities --------------------------------------------------------
        if tradition_method == 'CNM':
            communities = nx.algorithms.community.greedy_modularity_communities(g, weight=weight, resolution=1.72, cutoff=1.2, best_n=None)
            trj_labels = []
            for c in communities:
                trj_labels.append(list(c))
            print('trj==', trj_labels)
            communities = trj_labels
            print('=====> 社区划分结果：', communities)
            cluster_num = len(communities)
            node_name_cluster_dict = {}
            cluster_point_dict = {}
            for (i, cluster) in enumerate(communities):
                cluster_point_dict[i] = list(cluster)
                for cluster_id in cluster:
                    node_name_cluster_dict[cluster_id] = i

           # MAGI方法 (2024年新方法) -------------------------------------------------------
        if use_magi:
            if use_line_graph:
                # 使用线图的特征和邻接矩阵，参考GCC成功经验调参
                trj_labels = run_magi(adj_mat, features, cluster_num,
                                      epochs=2000,  # 更多训练轮数
                                      lr=0.000001,  # 更小学习率
                                      modularity_weight=5.0,  # 极强调模块度
                                      contrastive_weight=2.0)  # 强调对比学习
                node_name_cluster_dict = {}
                cluster_point_dict = {}
                for i in range(len(trj_labels)):
                    label = int(trj_labels[i])
                    if label not in cluster_point_dict:
                        cluster_point_dict[label] = []
                    cluster_point_dict[label].append(related_node_names[i])
                    node_name_cluster_dict[related_node_names[i]] = label
                print('MAGI 社区发现结果: ', cluster_point_dict)
                print('实际有效社区个数: ', len(cluster_point_dict.keys()))
                exp5_log.append(f'MAGI实际有效社区个数: {get_ok_cluster_num(cluster_point_dict)}')
            else:
                # 使用原图进行MAGI聚类
                adj_mat = nx.adjacency_matrix(g)
                # 创建简单的节点特征（度特征 + 随机特征）
                degrees = dict(g.degree())
                node_list = list(g.nodes())
                features = []
                for node in node_list:
                    # 使用度作为基础特征，添加一些随机特征
                    feat = [degrees[node]] + [np.random.random() for _ in range(9)]  # 10维特征
                    features.append(feat)
                features = np.array(features)
                
                # 调整参数以获得更好的聚类效果
                trj_labels = run_magi(adj_mat, features, cluster_num,
                                    epochs=500,      # 增加训练轮数
                                    lr=0.0001)       # 降低学习率
                node_name_cluster_dict = {}
                cluster_point_dict = {}
                for i, node in enumerate(node_list):
                    label = int(trj_labels[i])
                    if label not in cluster_point_dict:
                        cluster_point_dict[label] = []
                    cluster_point_dict[label].append(node)
                    node_name_cluster_dict[node] = label
                print('MAGI 社区发现结果: ', cluster_point_dict)
                
        # 原有的GCC方法 ----------------------------------------------------------------------
        elif use_line_graph:
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
            print('GCC实际有效社区个数: ', get_ok_cluster_num(cluster_point_dict))
            exp5_log.append(f'GCC实际有效社区个数: {get_ok_cluster_num(cluster_point_dict)}')
        # print(
        #     f'=========> feat len={len(features)}  nodename len={len(related_node_names)}  label len={len(trj_labels)}')
        # print(list(trj_labels))
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
        # print(len(g.nodes))

        # print(f'====> 社区个数：{cluster_num}, Q = {Q(lg, node_name_cluster_dict)}')
        print(f'====> 社区个数：{cluster_num}, CON = {avg_CON(g, cluster_point_dict, node_name_cluster_dict, use_igraph)}')
        # print(f'====> 社区个数：{cluster_num}, TPR = {avg_TPR(lg, cluster_point_dict)}')

    file_name = f'{exp5_log_name}.txt'
    f = open(file_name, 'w')
    for log in exp5_log:
        f.write(log + '\n')
    f.close()


if __name__ == '__main__':
    def print_time():  # 无限循环
        print('-------------------->>>>  start')
        while True:  # 获取当前的时间
            current_time = time.ctime(time.time())  # 输出线程的名字和时间
            print('keep live', current_time)  # 休眠10分钟，即600秒 time.sleep(600)
            time.sleep(600)

    # thread = threading.Thread(target=print_time)
    # thread.start()

    od_region = get_region()
    with open("./data/region.pkl", 'rb') as file:
        trj_region = pickle.loads(file.read())
    # makeVocab(trj_region, h5_files)
    total_od_pairs = get_od_filter_by_day_and_hour(month, start_day, end_day, start_hour, end_hour, od_region)

    # get_od_hot_cell 的后2个参数：1000 是只考虑当前区域和时间段内最热门的k=1000个OD对，lower_bound=0是过滤阈值，即流量大于0的OD都会被加入数据集
    od_pairs, od_cell_set, od_pair_set, hot_od_gps_set, od_flow_dict = get_od_hot_cell(total_od_pairs, od_region, 1000, 1)
    res = get_grid_split(od_region, od_pair_set, hot_od_gps_set)
    get_line_graph(od_region, trj_region, month, start_day, end_day, start_hour, end_hour, res['out_adj_table'],
                   res['cluster_point_dict'], od_pair_set)
