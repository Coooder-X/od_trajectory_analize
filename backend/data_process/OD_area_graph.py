# get_trip_endpoints_filter_by_coords
import json
import math
import pickle
from datetime import datetime

import numpy as np

from data_process.DT_graph_clustering import get_data, od_points_filter_by_hour, delaunay_clustering, get_data_filter_by_coords, draw_DT_clusters
from poi_process.read_poi import lonlat2meters_coords
from vis.trajectoryVIS import FileInfo

from graph_process.Graph import Graph
from graph_process.Point import Point
from test import adj, cids

from data_process.od_pair_process import get_total_od_points


def build_od_graph(point_cluster_dict: dict, total_od_points, index_lst: list):
    """
    2 个作用：
    1、生成适用于力导向布局的 json 邻接表和节点数组
    2、生成 map<int, set<int>> 形式的邻接表，用于传给前端 store 使用（如刷选功能）
    :param point_cluster_dict: 根据时间过滤后的，当前时间段内的
    :param total_od_points [[lon, lat, time, trj_id, flag], [], ...] trj_id 是轨迹id，flag==0 表示 O 点，1表示 D 点
    :param index_lst: od_points 【全量】 od 点对应的索引
    :return json_adj_table: 元素形式为 {'source': st, 'target': ed} 的数组，st、ed 为簇索引
    :return json_nodes: 数组，每个元素形式为 {'name': i}, i 是 od 点在全量 od 点中的索引
    :return adj_table: 邻接表，map<int, set<int>> 形式的，存储簇索引标识的簇之间的邻接关系
    """
    out_adj_table = {}
    in_adj_table = {}
    for pid in index_lst:
        od = total_od_points[pid]  # 当前 od 点具体数据
        trj_id = od[4]  # 当前 od 是 O 还是 D 点
        if trj_id == 0:  # 如果当前 od 点是 O 点，则找到对应的 D 点所在簇，更新 出边邻接表
            src_cid = point_cluster_dict[pid]
            if pid + 1 >= len(total_od_points) or pid + 1 not in point_cluster_dict:  # 若不存在 pid+1 或这个 D 点所在簇不在当前时间段内出现，则跳过
                continue
            if src_cid not in out_adj_table:
                out_adj_table[src_cid] = set()
            out_adj_table[src_cid].add(point_cluster_dict[pid + 1])
        else:  # 如果当前 od 点是 D 点，则找到对应的 O 点所在簇，更新 入边邻接表
            tgt_cid = point_cluster_dict[pid]
            if pid - 1 < 0 or pid - 1 not in point_cluster_dict:  # 若不存在 pid-1 或这个 O 点所在簇不在当前时间段内出现，则跳过
                continue
            if tgt_cid not in in_adj_table:
                in_adj_table[tgt_cid] = set()
            in_adj_table[tgt_cid].add(point_cluster_dict[pid - 1])

    new_out_adj_table = {}
    new_in_adj_table = {}

    for k in out_adj_table:
        new_out_adj_table[k] = list(out_adj_table[k])
    for k in in_adj_table:
        new_in_adj_table[k] = list(in_adj_table[k])
    return new_out_adj_table, new_in_adj_table
    '''json_adj_table, new_json_nodes,'''


def get_line_graph_by_selected_cluster(selected_cluster_ids_in_brush, selected_cluster_ids, out_adj_dict, exp_od_pair_set):
    """
    :param selected_cluster_ids_in_brush: 一个数组，存储地图中选取框框内的簇的id
    :param selected_cluster_ids: 一个数组，存储地图中已选的所有簇的id
    :param out_adj_dict: 当天数据中所有簇的全量的邻接表，out_adj_dict[x] 存储簇 id 为 x 的簇，会到达的簇的 id 数组
    :exp_od_pair_set: 一个set，包含一些OD对，只有在这个集合中的OD对才可以被用于建图
    :return force_nodes: 转换成的线图的节点数组
    :return force_edges: 转换成的线图的边数组
    :return filtered_adj_dict: 根据已选簇，从全量簇的邻接表中过滤出的已选簇的邻接表
    :return line_graph: 转换后的线图，为 networkx 对象
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
    line_graph = g.getLineGraph()
    # g.drawGraph()
    # g.drawLineGraph()
    # print(line_graph.nodes)
    # print(line_graph.edges)
    # print('点数据', line_graph.nodes.data())
    print('点个数', len(line_graph.nodes))
    print('边个数', len(line_graph.edges))

    force_nodes = []
    for node in line_graph.nodes:
        force_nodes.append({ 'name': f'{node[0]}_{node[1]}' })
    force_edges = []
    for edge in line_graph.edges:
        p1, p2 = edge[0], edge[1]
        force_edges.append({ 'source': f'{p1[0]}_{p1[1]}', 'target': f'{p2[0]}_{p2[1]}' })
    return force_nodes, force_edges, filtered_adj_dict, line_graph


# def aggregate_single_points(force_nodes, force_edges, filtered_adj_dict):
#     """
#     :param with_space_dict: 是否考虑空间距离，如果不考虑，则把离散的点都连上 fake 边，保证它们聚集在一块，不会扩散到整个屏幕
#     """
#     single_points_names = []
#     for node in force_nodes:
#         flag = False
#         for edge in force_edges:
#             if edge['source'] == node['name'] or edge['target'] == node['name']:
#                 flag = True
#                 break
#         if not flag:
#             single_points_names.append(node['name'])
#
#         # src_cid, tgt_cid = list(map(int, node['name'].split('_')))
#         # if len(filtered_adj_dict[tgt_cid]) > 0:
#         #     continue
#         # flag = True
#         # for cid in filtered_adj_dict:
#         #     if src_cid in filtered_adj_dict[cid]:
#         #         flag = False
#         #         break
#         # if flag:
#         #     continue
#         # single_points_names.append(node['name'])
#
#     print('离散点个数：', len(single_points_names))
#
#     for i in range(len(single_points_names)):
#         for j in range(i+1, len(single_points_names)):
#             src_name, tgt_name = single_points_names[i], single_points_names[j]
#             force_edges.append({ 'source': f'{src_name}', 'target': f'{tgt_name}', 'singleFake': True })
#     return force_edges


def get_cluster_center_coord(cluster_point_dict: dict, selected_cluster_idxs: list):
    """
    :param cluster_point_dict 簇id 到 OD点 id数组 的映射
    :param selected_cluster_idxs: 选择到的所有簇的 id
    :return cid_center_coord_dict: Map<簇id, [lon, lat]> 的映射，存储簇中心点的坐标
    """
    cid_center_coord_dict = {}
    # total_od_points [[lon, lat, time, trj_id, flag], [], ...] trj_id 是轨迹id，flag==0 表示 O 点，1表示 D 点
    total_od_points = get_total_od_points()  # 当天全量的 od 点数据 TODO：于淼的数据处理完成后，从那个接口拿

    for cid in selected_cluster_idxs:
        p_idxs = cluster_point_dict[cid]  # 这里是根据全量的簇内点计算的中心点，而不是框选出的当前时间段的！！！
        n = len(p_idxs)
        avg_lon, avg_lat = 0, 0
        for pid in p_idxs:
            lon, lat = total_od_points[pid][0], total_od_points[pid][1]
            avg_lon += lon / n
            avg_lat += lat / n
        cid_center_coord_dict[cid] = [avg_lon, avg_lat]

    return cid_center_coord_dict


def fuse_fake_edge_into_linegraph(force_nodes, force_edges, edge_name_dist_map: dict):
    """
    输入原始线图，给其添加 fake 边，并给所有边加入距离属性 'value'
    :param force_nodes: 原始线图的节点
    :param force_edges: 原始线图的边
    :param edge_name_dist_map: 经过计算的 OD 对距离map，形如 map<'12_34-45_78', dist>，key 是用'-'隔开的两个OD对的name
    :return force_edges: 修改过后的线图的边，结构适用于 d3 力导向。由于节点不变所以不返回
    """
    exist_edge = set()
    # 记录原线图中本就存在的边，以及给这些边添加距离属性，即 ‘value’
    for edge in force_edges:
        src, tgt = edge['source'], edge['target']
        exist_edge.add(f'{src}-{tgt}')
        exist_edge.add(f'{tgt}-{src}')
        if f'{src}-{tgt}' in edge_name_dist_map:
            edge['value'] = math.floor(edge_name_dist_map[f'{src}-{tgt}'])
        elif f'{tgt}-{src}' in edge_name_dist_map:
            edge['value'] = math.floor(edge_name_dist_map[f'{tgt}-{src}'])

    # 给原线图添加 fake 边和距离属性
    for i in range(len(force_nodes)):
        od_pair1 = force_nodes[i]['name']
        for j in range(i+1, len(force_nodes)):
            od_pair2 = force_nodes[j]['name']
            possible_edge_name1 = f'{od_pair1}-{od_pair2}'
            possible_edge_name2 = f'{od_pair2}-{od_pair1}'
            if possible_edge_name1 in exist_edge or possible_edge_name2 in exist_edge:  # 已存在原线图中的边就跳过
                continue
            # 两个点只添加一条单向边就行了
            force_edges.append({
                'source': f'{od_pair1}',
                'target': f'{od_pair2}',
                'value': math.floor(edge_name_dist_map[f'{od_pair1}-{od_pair2}']),
                'isFake': True,
            })
    # print('new force_edges', force_edges)
    return force_edges


def get_cluster_center_coord(total_od_points, cluster_point_dict: dict, selected_cluster_idxs: list):
    """
    :param total_od_points: 全量 od 点数据
    :param cluster_point_dict 簇id 到 OD点 id数组 的映射
    :param selected_cluster_idxs: 选择到的所有簇的 id
    :return cid_center_coord_dict: Map<簇id, [lon, lat]> 的映射，存储簇中心点的坐标
    """
    cid_center_coord_dict = {}
    # total_od_points [[lon, lat, time, trj_id, flag], [], ...] trj_id 是轨迹id，flag==0 表示 O 点，1表示 D 点
    # total_od_points = get_total_od_points()  # 当天全量的 od 点数据 TODO：于淼的数据处理完成后，从那个接口拿

    for cid in selected_cluster_idxs:
        p_idxs = cluster_point_dict[cid]  # 这里是根据全量的簇内点计算的中心点，而不是框选出的当前时间段的！！！
        n = len(p_idxs)
        avg_lon, avg_lat = 0, 0
        for pid in p_idxs:
            lon, lat = total_od_points[pid][0], total_od_points[pid][1]
            avg_lon += lon / n
            avg_lat += lat / n
        cid_center_coord_dict[cid] = [avg_lon, avg_lat]

    return cid_center_coord_dict


if __name__ == '__main__':
    k, theta = 8, 10
    print('开始读取OD点')
    start_time = datetime.now()
    od_points = np.asarray(lonlat2meters_coords(coords=get_data(), use_time=True))

    total_od_coord_points = od_points[:, 0:2]  # 并去掉时间戳留下经纬度坐标
    print('读取OD点结束，用时: ', (datetime.now() - start_time))
    print('pos nums', len(od_points), '\n开始聚类')
    start_time = datetime.now()
    point_cluster_dict, cluster_point_dict = delaunay_clustering(k=k, theta=theta, od_points=total_od_coord_points)
    end_time = datetime.now()
    print('结束聚类，用时: ', (datetime.now() - start_time))
    # draw_DT_clusters(cluster_point_dict, od_points, k, theta, start_hour, end_hour)
    # draw_time = datetime.now()
    # print('画图用时: ', draw_time - end_time)

    start_hour, end_hour = 10, 15
    (part_od_coord_points, index_lst) = od_points_filter_by_hour(od_points, start_hour, end_hour)  # 过滤出所有在该时间段的 od 点
    index_lst = index_lst[0]
    print(index_lst)
    json_adj_table, json_links, json_nodes = build_od_graph(point_cluster_dict, od_points, index_lst)
    print(json_links)
    print(json_nodes)
    # part_od_coord_points = part_od_coord_points[:, 0:2]  # 并去掉时间戳留下坐标
    # draw_DT_clusters(cluster_point_dict, total_od_coord_points, k, theta, start_hour, end_hour, set(index_lst.tolist()))
    # draw_time = datetime.now()
    # print('画图用时: ', draw_time - end_time)

    # area_od_points = np.asarray(lonlat2meters_coords(coords=get_data_filter_by_coords((120.064491, 30.29897), (120.194491, 30.339897)), use_time=True))
    # area_od_points = od_points_filter_by_hour(area_od_points, start_hour, end_hour)[:, 0:2]  # 过滤出所有在该时间段的 od 点，并去掉时间戳留下坐标
    # end_time =lll datetime.now()
    # print('结束聚类，用时: ', (datetime.now() - start_time))
    # draw_DT_clusters(cluster_point_dict, od_points, k, theta, start_hour, end_hour)
    # draw_time = datetime.now()
    # print('画图用时: ', draw_time - end_time)


