import json
import os
import pickle
import random
import numpy as np
from matplotlib import pyplot as plt

from data_process.SpatialRegionTools import gps2cell
from graph_process.Graph import Graph
from graph_process.Point import Point
from poi_process.read_poi import buildKDTree, getPOI_Coor, lonlat2meters_poi
from trip_process.read_trips import getTrips
from utils import lonlat2meters, cal_meter_dist
from vis.trajectoryVIS import FileInfo, randomcolor, draw_points

# def is_connect(region, kdtree, points, radius):

'''
尝试只用半径范围判断两点之间的连通性
'''
# def build_graph(kdtree, total_points, input_points, trip_index, radius):
#     g = Graph()
#     result = kdtree.query_ball_point(input_points, radius)
#     points = np.asarray(total_points)
#     print('result', result)
#     for i, res in enumerate(result):
#         print('res', res)
#         nearby_points = points[res]
#         print('nearby_points', nearby_points)
#         from_point_id = trip_index[i]
#         from_point = Point(str(from_point_id), from_point_id, from_point_id, {total_points[from_point_id]})
#         g.addVertex(from_point)
#         for to_point_id in res:
#             to_point = Point(str(to_point_id), to_point_id, to_point_id, {total_points[to_point_id]})
#             g.addVertex(to_point)
#             g.addDirectLine(from_point, [to_point, ])


'''
:param trj_point_kdtree     （米为单位的轨迹端点建立的 kdtree）
:param poi_kdtree           （米为单位的 POI 坐标建立的 kdtree）
:param trj_points           （米为单位的轨迹端点坐标 list）
:param poi                  （米为单位的 POI 坐标 list）
:param trj_point_vis        （轨迹端点是否访问的标记 list）
:param end2poi_dict         （轨迹端点到其对应的代表 POI 的映射）
:param radius               （米为单位的以轨迹端点为中心的查找半径）
'''
def point_compression(trj_point_kdtree, poi_kdtree, trj_points, poi, trj_point_vis, end2poi_dict, radius):
    g = Graph()
    for i, endpoint in enumerate(trj_points):
        if trj_point_vis[i]:  # 遍历过程中，排除已经遍历过的轨迹端点
            continue
        possible_idx_set = trj_point_kdtree.query_ball_point(endpoint, radius)  # 当前点半径radius范围内的所有轨迹端点id，包括自己
        # print('possible_set', possible_idx_set)
        # print(len(possible_idx_set), len(trj_points))
        possible_set = []  # 当前点半径radius范围内的所有轨迹端点列表（可能与当前求出的代表 POI 相连的轨迹端点）
        possible_poi_set = []  # 可能成为代表 POI 的 POI 节点
        count_dict = {}
        for possible_index in possible_idx_set:
            if trj_point_vis[possible_index]:
                continue
            possible_set.append(trj_points[possible_index])
            possible_poi = poi_kdtree.query_ball_point(trj_points[possible_index], radius)  # 找当前轨迹端点半径radius范围内的所有POI
            possible_poi_set.append(possible_poi)
            # print('possible_poi', len(possible_poi), len(poi))
            for idx in possible_poi:
                if idx not in count_dict.keys():
                    count_dict[idx] = 1
                else:
                    count_dict[idx] += 1
        items = list(count_dict.items())
        items.sort(key=lambda x: -x[1])
        if len(items) == 0:
            print('当前点附近找不到可映射的POI')
            continue
        symbol_poi_idx = items[0][0]  # 求出了当前区域内的代表 POI 节点的 id
        symbol_point = Point(str(symbol_poi_idx), symbol_poi_idx, symbol_poi_idx, {})
        # if random.random() < 0.9:  #  防止图太大，过滤一部分点
        #     continue
        g.addVertex(symbol_point)  # 作为原图的点加入图中
        # 将范围内的轨迹端点与代表 POI 做映射，TODO：注意处理自环的情况
        for possible_index in possible_idx_set:
            if trj_point_vis[possible_index]:
                continue
            if cal_meter_dist(trj_points[possible_index], poi[symbol_poi_idx]) < radius:
                end2poi_dict[possible_index] = symbol_poi_idx
                trj_point_vis[possible_index] = True
        # TODO: 处理vis为 false 的点(没有映射到poi的轨迹端点)

    print('新图中的点数：', len(g.nodeList))
    cnt = 0
    for p in trj_point_vis:
        if not p:
            cnt += 1
    print('未映射到POI的轨迹端点数：', cnt, '总点数：', len(trj_points))
    poiidx2point_dict = {}
    for point in g.nodeList:
        poiidx2point_dict[point.nodeId] = point
    return g, end2poi_dict, poiidx2point_dict


def point_compression_with_cell(endpoints):
    cell_color_dict = {}
    point_color_dict = {}
    with open("../data/region.pkl", 'rb') as file:
        region = pickle.loads(file.read())
    for i, point in enumerate(endpoints):
        cell = gps2cell(region, point[0], point[1])
        if cell not in cell_color_dict.keys():
            color = randomcolor()
            cell_color_dict[cell] = color
            point_color_dict[i] = color
        else:
            point_color_dict[i] = cell_color_dict[cell]
    return point_color_dict


def draw_point_compression_with_cell(endpoints, point_color_dict):
    fig = plt.figure(figsize=(20, 10))
    ax = fig.subplots()
    print('开始绘图')
    for i, point in enumerate(endpoints):
        ax.scatter(point[0], point[1], c=point_color_dict[i], marker='o', s=4)
    ax.set_xlabel('lon')  # 画出坐标轴
    ax.set_ylabel('lat')
    plt.show()


def build_graph(g, end2poi_dict, poiidx2point_dict, trips):
    # TODO: 按一定规则过滤一下节点和边，使得可视化结果有可读性
    for i, trip in enumerate(trips):
        time1, time2 = trip[0][2], trip[-1][2]
        if time1 < time2:
            start_idx, end_idx = 2 * i, 2 * i + 1
        else:
            start_idx, end_idx = 2 * i + 1, 2 * i
        if start_idx not in end2poi_dict or end_idx not in end2poi_dict:
            continue
        # if random.random() < 0.2:  #  防止图太大，过滤一部分边
        #     g.addDirectLine(poiidx2point_dict[end2poi_dict[start_idx]], [[poiidx2point_dict[end2poi_dict[end_idx]], 1]])  # 边权暂时设置为1 TODO：边权用轨迹表示代替
        g.addDirectLine(poiidx2point_dict[end2poi_dict[start_idx]], [[poiidx2point_dict[end2poi_dict[end_idx]], 1]])  # 边权暂时设置为1 TODO：边权用轨迹表示代替
    return g


if __name__ == "__main__":
    fileInfo = FileInfo()
    with open("../conf/graph_gen.json") as conf:
        # 读取配置文件
        json_data = json.load(conf)
        fileInfo.trj_data_path = json_data['trj_data_path']
        fileInfo.trj_data_date = json_data['trj_data_date']
        fileInfo.trj_file_name = json_data['trj_file_name']
        fileInfo.poi_dir = json_data['poi_dir']
        fileInfo.poi_file_name_lst = os.listdir(fileInfo.poi_dir)  # json_data['poi_file_name_lst']

    filter_step = 10
    use_cell = False
    trips, lines = getTrips(fileInfo, filter_step, use_cell)
    trip_index = [i for i in range(len(trips))]
    endpoints = []
    for trip in trips:  # 将轨迹起点和终点转换成米单位的坐标，存入kdtree，便于通过半径查找点
        x, y = lonlat2meters(trip[0][0], trip[0][1])
        endpoints.append([x, y])
        x, y = lonlat2meters(trip[-1][0], trip[-1][1])
        endpoints.append([x, y])
    # todo: endpoints 按空间排序
    endpoints = sorted(endpoints, key=lambda x: (x[0], x[1]))
    trj_point_kdtree = buildKDTree(endpoints)

    poi = getPOI_Coor(fileInfo.poi_dir, fileInfo.poi_file_name_lst)
    poi = lonlat2meters_poi(poi)
    poi_kdtree = buildKDTree(poi)
    trj_point_vis = [False for i in range(len(endpoints))]
    end2poi_dict = {}
    radius = 700
    g, end2poi_dict, poiidx2point_dict = point_compression(trj_point_kdtree, poi_kdtree, endpoints, poi, trj_point_vis, end2poi_dict, radius)
    # g = build_graph(g, end2poi_dict, poiidx2point_dict, trips)
    # g.drawGraph()
    # L = g.drawLineGraph()

    # ============== 可以将缩点行为看作聚类，绘制缩点后的轨迹端点的簇分布=====================================
    # print('end2poi_dict', end2poi_dict)
    color_dict = {}
    fig = plt.figure(figsize=(20, 10))
    ax = fig.subplots()
    print('开始绘图')
    for end_p in end2poi_dict.keys():
        if end2poi_dict[end_p] in color_dict.keys():
            pass
        else:
            color_dict[end2poi_dict[end_p]] = randomcolor()
        ax.scatter(endpoints[end_p][0], endpoints[end_p][1], c=color_dict[end2poi_dict[end_p]], marker='o', s=4)
    print('结束绘图，开始保存')
    ax.set_xlabel('lon')  # 画出坐标轴
    ax.set_ylabel('lat')
    plt.savefig(f'../../figs/缩点算法轨迹端点聚类_r={radius}.png', dpi=300)
    plt.show()
    draw_points(endpoints, 'g')
    #===============================================================================================

    #+++++++ 不用算法缩点，而是把处于一个cell的轨迹端点看作属于一个簇 ++++++++++++++++++++++++++++++++++++++
    # point_color_dict = point_compression_with_cell(endpoints)
    # print(len(endpoints), len(point_color_dict.keys()))
    # draw_point_compression_with_cell(endpoints, point_color_dict)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # 可以将缩点行为看作聚类，绘制缩点后的轨迹端点的簇分布
    print('end2poi_dict', end2poi_dict)
    color_dict = {}
    fig = plt.figure(figsize=(20, 10))
    ax = fig.subplots()
    for end_p in end2poi_dict.keys():
        if end2poi_dict[end_p] in color_dict.keys():
            pass
        else:
            color_dict[end2poi_dict[end_p]] = randomcolor()
        ax.scatter(endpoints[end_p][0], endpoints[end_p][1], c=color_dict[end2poi_dict[end_p]], marker='o', s=4)

    ax.set_xlabel('lon')  # 画出坐标轴
    ax.set_ylabel('lat')
    plt.show()
    draw_points(endpoints, 'g')

    # for trip in trips: # 将轨迹起点和终点转换成米单位的坐标，分别存入kdtree，便于通过半径查找点
    #     time1, time2 = trip[0][2], trip[-1][2]
    #     x1, y1 = lonlat2meters(trip[0][0], trip[0][1])
    #     p1 = [x1, y1]
    #     x2, y2 = lonlat2meters(trip[-1][0], trip[-1][1])
    #     p2 = [x2, y2]
    #     if time1 < time2:
    #         start_points.append(p1)
    #         end_points.append(p2)
    #     else:
    #         start_points.append(p2)
    #         end_points.append(p1)
    # start_kdtree = buildKDTree(start_points)
    # end_kdtree = buildKDTree(end_points)

    # radius = 2000
    # build_graph(start_kdtree, start_points, end_points, trip_index, radius)
