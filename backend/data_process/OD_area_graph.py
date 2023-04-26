# get_trip_endpoints_filter_by_coords
import json
import pickle
from datetime import datetime

import numpy as np

from data_process.DT_graph_clustering import get_data, od_points_filter_by_hour, delaunay_clustering, get_data_filter_by_coords, draw_DT_clusters
from poi_process.read_poi import lonlat2meters_coords
from vis.trajectoryVIS import FileInfo

from graph_process.Graph import Graph
from graph_process.Point import Point
from test import adj, cids


def build_od_graph(point_cluster_dict: dict, total_od_points, index_lst: list):
    """
    2 个作用：
    1、生成适用于力导向布局的 json 邻接表和节点数组
    2、生成 map<int, set<int>> 形式的邻接表，用于传给前端 store 使用（如刷选功能）
    :param point_cluster_dict
    :param total_od_points [[lon, lat, time, trj_id, flag], [], ...] trj_id 是轨迹id，flag==0 表示 O 点，1表示 D 点
    :param index_lst: od_points 【全量】 od 点对应的索引
    :return json_adj_table: 元素形式为 {'source': st, 'target': ed} 的数组，st、ed 为簇索引
    :return json_nodes: 数组，每个元素形式为 {'name': i}, i 是 od 点在全量 od 点中的索引
    :return adj_table: 邻接表，map<int, set<int>> 形式的，存储簇索引标识的簇之间的邻接关系
    """
    json_adj_table = []
    json_nodes = set()
    out_adj_table = {}
    in_adj_table = {}
    for i in index_lst:
        json_nodes.add(point_cluster_dict[i])
        # json_nodes.append({'name': i})
        for j in index_lst:
            cid_i, cid_j = point_cluster_dict[i], point_cluster_dict[j]
            if total_od_points[i][3] == total_od_points[j][3] and cid_i != cid_j:
                (st, ed) = (cid_i, cid_j) if total_od_points[i][4] == 0 else (cid_j, cid_i)
                json_adj_table.append({'source': st, 'target': ed})
                if st not in out_adj_table:
                    out_adj_table[st] = set()
                out_adj_table[st].add(ed)
                if ed not in in_adj_table:
                    in_adj_table[ed] = set()
                in_adj_table[ed].add(st)
    new_out_adj_table = {}
    new_in_adj_table = {}
    new_json_nodes = []
    for cid in json_nodes:
        new_json_nodes.append({'id': cid})
    for k in out_adj_table:
        new_out_adj_table[k] = list(out_adj_table[k])
    for k in in_adj_table:
        new_in_adj_table[k] = list(in_adj_table[k])
    return json_adj_table, new_json_nodes, new_out_adj_table, new_in_adj_table


def get_line_graph_by_selected_cluster(selected_cluster_ids, out_adj_dict):
    """
    :param selected_cluster_ids: 一个数组，存储地图中已选的所有簇的id
    :param out_adj_dict: 当天数据中所有簇的全量的邻接表，out_adj_dict[x] 存储簇 id 为 x 的簇，会到达的簇的 id 数组
    :return force_nodes: 转换成的线图的节点数组
    :return force_edges: 转换成的线图的边数组
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
            if cid in out_adj_dict and to_cid in out_adj_dict[cid]:
                print(cid, to_cid)
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
    print(line_graph.nodes)
    print(line_graph.edges)
    print('点个数', len(line_graph.nodes))
    print('边个数', len(line_graph.edges))

    force_nodes = []
    for node in line_graph.nodes:
        force_nodes.append({ 'name': f'{node[0]}_{node[1]}' })
    force_edges = []
    for edge in line_graph.edges:
        p1, p2 = edge[0], edge[1]
        force_edges.append({ 'source': f'{p1[0]}_{p1[1]}', 'target': f'{p2[0]}_{p2[1]}' })
    return force_nodes, force_edges, filtered_adj_dict


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


