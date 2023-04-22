# get_trip_endpoints_filter_by_coords
import json
import pickle
from datetime import datetime

import numpy as np

from data_process.DT_graph_clustering import get_data, od_points_filter_by_hour, delaunay_clustering, get_data_filter_by_coords, draw_DT_clusters
from poi_process.read_poi import lonlat2meters_coords
from vis.trajectoryVIS import FileInfo


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
    json_nodes = []
    out_adj_table = {}
    in_adj_table = {}
    for i in index_lst:
        json_nodes.append({'name': i})
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
    print(out_adj_table)
    for k in out_adj_table:
        new_out_adj_table[k] = list(out_adj_table[k])
    for k in in_adj_table:
        new_in_adj_table[k] = list(in_adj_table[k])
    return json_adj_table, json_nodes, new_out_adj_table, new_in_adj_table


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


