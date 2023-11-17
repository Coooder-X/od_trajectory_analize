import os
import pickle
from datetime import datetime
import time
import threading

import numpy as np
import torch

import utils
from data_process import od_pair_process
from data_process.OD_area_graph import build_od_graph, fuse_fake_edge_into_linegraph, \
    get_line_graph_by_selected_cluster
from data_process.SpatialRegionTools import get_cell_id_center_coord_dict, makeVocab, inregionS
from data_process.od_pair_process import get_trips_by_ids, get_trj_ids_by_force_node, \
    get_odpair_space_similarity
from data_process.spatial_grid_utils import get_region, get_od_points_filter_by_region, divide_od_into_grid
from gcc.graph_convolutional_clustering.gcc.run import run, draw_cluster_in_trj_view, draw_cluster_in_trj_view_new
from graph_process.Graph import get_degree_by_node_name, get_feature_list, get_adj_matrix
from t2vec import args
from t2vec_graph import run_model2, get_cluster_by_trj_feature
import networkx as nx


def Q(G, node_name_cluster_dict):
    """
    @node_name_cluster_dict: 节点名到社区id的映射
    """
    node_names = set(list(node_name_cluster_dict.keys()))
    m = len(G.edges())
    res = 0.0
    # for a in G.nodes():
    #     src, tgt = G.nodes[a]['name'].split('-')
    #     a_name = f'{src}_{tgt}'
    #     for b in G.nodes():
    #         if a == b:
    #             continue
    #         src, tgt = G.nodes[b]['name'].split('-')
    #         b_name = f'{src}_{tgt}'
    #         Aab = 1 if (a, b, 0) in G.edges() else 0
    #         ksy = 0
    #         if a_name in node_names and b_name in node_names:
    #             ksy = 1 if node_name_cluster_dict[a_name] == node_name_cluster_dict[b_name] else 0
    #         # E = G.out_degree(a) * G.out_degree(b) / m
    #         # E = G.degree(a) * G.degree(b) / (2 * m)
    #         E = G.out_degree(a) * G.in_degree(b) / m
    #         res += (Aab - E) * ksy / m
    #         if (Aab - E) * ksy / m > 0:
    #             print(f'Aab={Aab}, E={E}, ksy={ksy}, cur={(Aab - E) * ksy / m}')

    labels = list(node_name_cluster_dict.values())
    for cluster_id in labels:
        ai = 0
        eij = 0
        for a in G.nodes():
            src, tgt = G.nodes[a]['name'].split('-')
            a_name = f'{src}_{tgt}'
            ksy = 1 if node_name_cluster_dict[a_name] == cluster_id else 0
            ai += (1 / m) * G.out_degree(a) * ksy
            for b in G.nodes():
                src, tgt = G.nodes[b]['name'].split('-')
                b_name = f'{src}_{tgt}'
                Aab = 1 if (a, b, 0) in G.edges() else 0
                ksy = node_name_cluster_dict[a_name] == cluster_id and node_name_cluster_dict[b_name] == cluster_id
                eij += (1 / m) * Aab * ksy
        res += eij - ai ** 2

    return res


def CON(G, cluster_id, node_name_cluster_dict):
    start = datetime.now()
    m = len(G.edges())
    fz = 0
    for edge in G.edges():
        u, v = edge[0], edge[1]
        u_name, v_name = f'{u[0]}_{u[1]}', f'{v[0]}_{v[1]}'
        if (node_name_cluster_dict[u_name] == cluster_id and node_name_cluster_dict[v_name] != cluster_id) or \
            (node_name_cluster_dict[u_name] != cluster_id and node_name_cluster_dict[v_name] == cluster_id):
            fz += 1
    vol_C = vol(G, cluster_id, node_name_cluster_dict)
    print(f'vol_C={vol_C}({cluster_id})')
    fm = min(vol_C, 2 * m - vol_C)
    fm = max(fm, 0.01)
    end = datetime.now()
    print('用时', end - start)
    res = fz / (fz + vol_C)
    print(f'CON=({res})')
    return res


def vol(G, cluster_id, node_name_cluster_dict):
    res = 0
    for node in G.nodes():
        src, tgt = G.nodes[node]['name'].split('-')
        name = f'{src}_{tgt}'
        if node_name_cluster_dict[name] == cluster_id:
            res += G.out_degree(node)
    return res


def avg_CON(G, cluster_point_dict, node_name_cluster_dict):
    avg = 0.0
    i = 0
    print(len(cluster_point_dict.keys()))
    for cluster_id in cluster_point_dict:
        i += 1
        print(f'cluster: {cluster_id}')
        avg += CON(G, cluster_id, node_name_cluster_dict)
    avg /= len(cluster_point_dict.keys())
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


def avg_TPR(G, cluster_point_dict):
    n = len(cluster_point_dict.keys())
    name_node_dict = {}
    for node in G.nodes():
        src, tgt = G.nodes[node]['name'].split('-')
        name = f'{src}_{tgt}'
        name_node_dict[name] = node

    avg = 0.0
    for cluster_id in cluster_point_dict:
        print(f'cluster: {cluster_id}')
        avg += TPR(G, cluster_id, cluster_point_dict, name_node_dict)
    avg /= n
    return avg


month = 5
start_day, end_day = 2, 2
start_hour, end_hour = 8, 9

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


def get_line_graph(region, trj_region, month, start_day, end_day, start_hour, end_hour, out_adj_table, cluster_point_dict):
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
    force_nodes, force_edges, filtered_adj_dict, lg = get_line_graph_by_selected_cluster(selected_cluster_ids, selected_cluster_ids, out_adj_table)

    # print('边 ', lg.edges())
    # print('点 ', lg.nodes())

    #  计算簇中心坐标 ========================================
    tmp = {}
    for key in cluster_point_dict:
        if int(key) in used_od_cells:
            tmp[int(key)] = cluster_point_dict[key]
    cluster_point_dict = tmp

    # total_od_points = cache.get('total_od_points')
    total_od_points = od_pair_process.get_od_points_filter_by_day_and_hour(month, start_day, end_day, 0, 24)['od_points']
    # cid_center_coord_dict = get_cluster_center_coord(total_od_points, cluster_point_dict, selected_cluster_ids)

    #  计算 OD 对之间的距离 ==================================
    # edge_name_dist_map = get_odpair_space_similarity(selected_cluster_ids, cid_center_coord_dict, force_nodes)

    #  将线图加入 fake 边，并给边添加距离，成为考虑 OD 对空间关系的线图 ===============================
    # if with_space_dist:
    # force_edges = fuse_fake_edge_into_linegraph(force_nodes, force_edges, edge_name_dist_map)
    #
    # print('完全线图-点数', len(force_nodes))
    # print('完全线图-边数', len(force_edges))

    # if not with_space_dist:
    #     force_edges = aggregate_single_points(force_nodes, force_edges, filtered_adj_dict)
    #     print('单独点聚合后-边数', len(force_edges))

    # +++++++++++++++ 轨迹获取和特征 ++++++++++++++
    # node_label_dict = None
    # if os.path.exists(f'./read_trjs_{start_day}_{end_day}_{start_hour}_{end_hour}.pkl'):
    #     with open(f'./read_trjs_{start_day}_{end_day}_{start_hour}_{end_hour}.pkl', 'rb') as f:
    #         obj = pickle.loads(f.read())
    #         trj_idxs, node_names_trjId_dict = obj['trj_idxs'], obj['node_names_trjId_dict']
    # else:
    trj_idxs, node_names_trjId_dict = get_trj_ids_by_force_node(force_nodes, cluster_point_dict, total_od_points, region)
    with open(f'./read_trjs_{start_day}_{end_day}_{start_hour}_{end_hour}.pkl', 'wb') as f:
        picklestring = pickle.dumps({
            'trj_idxs': trj_idxs,
            'node_names_trjId_dict': node_names_trjId_dict
        })
        f.write(picklestring)
        f.close()
    print('get_trj_ids_by_force_node')
    # ----------- 简单聚合，每个OD对取一个轨迹的特征---------
    # tmp_trj_idx = []
    # tmp_node_names = []
    # name_set = {}
    # for i in range(len(node_names)):
    #     if node_names[i] not in name_set:
    #         tmp_trj_idx.append(trj_idxs[i])
    #         tmp_node_names.append(node_names[i])
    #         name_set[node_names[i]] = 1
    # -------------------------------------------------
    # trj_idxs, node_names = tmp_trj_idx, tmp_node_names
    # print('按轨迹顺序排列的节点名的映射', node_names)
    # tid_trip_dict = get_trips_by_ids(trj_idxs, month, start_day, end_day)
    # gps_trips = get_trips_by_ids(trj_idxs, month, start_day, end_day)
    # print('=============> gps_trips', gps_trips)
    if os.path.isfile(args.best_model):
        print("=> loading best_model '{}'".format(args.best_model))
        best_model = torch.load(args.best_model)


    node_names_trjFeats_dict = {}   # 节点名 -> 包含的轨迹特征数组的 map
    trjId_node_name_dict = {}   # 轨迹ID -> 所在的节点名的 map
    node_names_trj_dict = {}    # 节点名 -> gps 轨迹数组的 map
    for node_name in node_names_trjId_dict:
        node_trj_idxs = node_names_trjId_dict[node_name]
        for trj_id in node_trj_idxs:
            trjId_node_name_dict[trj_id] = node_name

    trj_idxs = list(trjId_node_name_dict.keys())  # 所有轨迹id
    gps_trips = get_trips_by_ids(trj_idxs, month, start_day, end_day)
    for trip in gps_trips:
        o, d = trip[0], trip[-1]
        print('o在区域内', inregionS(region, o[0], o[1]))
        print('d在区域内', inregionS(region, d[0], d[1]))
    print('draw_cluster_in_trj_view======================')
    draw_cluster_in_trj_view([1 for i in range(len(gps_trips))], gps_trips)
    trj_feats = run_model2(args, gps_trips, best_model, trj_region)    # 特征数组，顺序与 trj_idxs 对应
    print(f'轨迹id数= {len(trj_idxs)}, 轨迹数 = {len(gps_trips)}, 特征数 = {len(trj_feats)}')

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


    # total_feats = []
    # for node_name in node_names_trjId_dict:
    #     node_trj_idxs = node_names_trjId_dict[node_name]
    #     print(f'====> 节点{node_name}的轨迹数量为', len(node_trj_idxs))
    #     node_gps_trips = get_trips_by_ids(node_trj_idxs, month, start_day, end_day)
    #     print('gps trips ===>', node_gps_trips)
    #     node_trj_feats = run_model2(args, node_gps_trips, best_model, trj_region)
    #     node_names_trjFeats_dict[node_name] = node_trj_feats
    #     total_feats.extend(node_trj_feats)
    # labels = get_cluster_by_trj_feature(args, total_feats)


    # gps_trips = list(tid_trip_dict.values())
    # feature = run_model2(args, gps_trips)
    # labels = get_cluster_by_trj_feature(args, feature)
    # print('labels', labels)
    # print('labels 数量', len(labels))
    # print('labels 全部类别数量', len(list(set(labels))))
    # print('labels 全部标签类别', list(set(labels)))
    # node_label_dict = {}
    # node_feature_dict = {}
    # trjid_feature_dict = {}
    # for i in range(len(labels)):
    #     node_label_dict[node_names[i]] = labels[i]
    # 节点名称（${cid1}_${cid2}）和对应OD对特征的映射关系
    # for i in range(len(feature)):
    #     node_feature_dict[node_names[i]] = feature[i]
    #     trjid_feature_dict[node_names[i]] = trj_idxs[i]
    # print(f'轨迹id数{len(trj_idxs)}  轨迹数： {len(gps_trips)}  特征数: {len(feature)}  字典大小：{len(node_feature_dict.keys())} {len(node_names)}')
    # +++++++++++++++ 轨迹获取和特征 ++++++++++++++

    # ============== GCC 社区发现代码 ===============
    adj_mat = get_adj_matrix(lg)  # 根据线图得到 csc稀疏矩阵类型的邻接矩阵
    features, related_node_names = get_feature_list(lg, node_names_trjFeats_dict, avg_num)  # 根据线图节点顺序，整理一个节点向量数组，以及对应顺序的node name

    # print(f'=====================> related_node_names={related_node_names}')
    # print(f'=====================> node_names_trj_dict={list(node_names_trj_dict.keys())}')
    # tmp_node_idxs = []
    #  得到features对应顺序的 节点名称 (这里 features 顺序就是节点顺序)
    # for node_name in related_node_names:
    #     tmp_node_idxs.append(trjid_feature_dict[node_name])
    print(f'线图节点个数：{len(lg.nodes())}, 向量个数：{len(features)}')
    print('向量长度', len(features[0]))

    ######## 仅在做实验时需要这个 for 循环，否则不需要循环，执行一次即可
    for cluster_num in [5, 10, 20, 30, 40]:
    # for cluster_num in [5, 50]:
        trj_labels = run(adj_mat, features, cluster_num)  # 得到社区划分结果，索引对应 features 的索引顺序，值是社区 id
        trj_labels = trj_labels.numpy().tolist()
        print(list(trj_labels))
        cluster_point_dict = {}
        node_name_cluster_dict = {}
        for i in range(len(trj_labels)):
            label = trj_labels[i]
            if label not in cluster_point_dict:
                cluster_point_dict[label] = []
            # 在线图中度为 0 的散点，视为噪声，从社区中排除
            if get_degree_by_node_name(lg, related_node_names[i]) > 0:
                cluster_point_dict[label].append(related_node_names[i])
                node_name_cluster_dict[related_node_names[i]] = label
        print('实际社区个数: ', len(cluster_point_dict.keys()))
        # dag_force_nodes, dag_force_edges = get_dag_from_community(cluster_point_dict, force_nodes)

        to_draw_trips_dict = {}
        for label in cluster_point_dict:
            to_draw_trips_dict[label] = []
            for node_name in cluster_point_dict[label]:
                to_draw_trips_dict[label].extend(node_names_trj_dict[node_name])
        # print('to_draw_trips_dict', to_draw_trips_dict)
        draw_cluster_in_trj_view_new(to_draw_trips_dict, cluster_num, region)
        tsne_points = utils.DoTSNE(features, 2, cluster_point_dict)
        print('tsne_points', len(tsne_points))
        print(len(lg.nodes))

        # print(f'====> 社区个数：{cluster_num}, Q = {Q(lg, node_name_cluster_dict)}')
        # print(f'====> 社区个数：{cluster_num}, CON = {avg_CON(lg, cluster_point_dict, node_name_cluster_dict)}')
        # print(f'====> 社区个数：{cluster_num}, TPR = {avg_TPR(lg, cluster_point_dict)}')

    return {
        'force_nodes': force_nodes,
        'force_edges': force_edges,
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
    with open("/home/zhengxuan.lin/project/deepcluster/data/region.pkl", 'rb') as file:
        trj_region = pickle.loads(file.read())
    # makeVocab(trj_region, h5_files)
    res = get_grid_split(od_region)
    get_line_graph(od_region, trj_region, month, start_day, end_day, start_hour, end_hour, res['out_adj_table'], res['cluster_point_dict'])

