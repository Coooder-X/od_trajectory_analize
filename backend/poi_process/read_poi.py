import json

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import spatial

from utils import lonlat2meters, meters2lonlat


# import POI


# ['名称', '大类', '中类', '小类', '地址', '省', '市', '区', 'WGS84_经度', 'WGS84_纬度']


def getPOI_Coor(data_dir, file_name_lst):
    # print(os.listdir(data_dir))
    print('当前要读取的 POI 文件: ', file_name_lst)
    file_list = os.listdir(data_dir)
    for i in range(len(file_list)):
        file_list[i] = data_dir + '/' + file_list[i]
        print(file_list[i])

    total_poi_coor = []
    for file_name in file_name_lst:
        df1 = pd.read_excel(data_dir + '/' + file_name)  # 读取xlsx中的第一个sheet
        poi_coor = np.asarray(df1.iloc[:, [-2, -1]].values)
        print(poi_coor)
        total_poi_coor.extend(poi_coor)
    return np.asarray(total_poi_coor)


"""
将所有POI坐标放入 KDTree，方便查找邻近点，从距离上判断连通性
:param: GPS 经纬度格式的POI数组
:return 存储POI坐标的 KDTree
"""
def buildKDTree(pois):
    kdtree = spatial.KDTree(pois)
    return kdtree


"""
将轨迹的头尾节点坐标放入 points，分别查找邻近点，从距离上判断连通性，返回通过轨迹头尾节点过滤后的poi数组
:param: trips：经纬度格式的轨迹数组，poi_coor：所有经纬度表示的POI节点，存储所有轨迹头尾节点经纬度坐标的 KDTree、最邻近的点数 k
:return coor_filtered：[[经度, 纬度], [lon, lat], ...] 形状的经过过滤的坐标数组
"""
def getPOI_filter_by_topk(trips, poi_coor, kdtree, k):
    points = []
    for trip in trips:
        points.append([trip[0][0], trip[0][1]])
        points.append([trip[-1][0], trip[-1][1]])
    topk_dist, topk_id = kdtree.query(points, k)
    coor_filtered = []
    for row in topk_id:
        if k == 1:
            coor_filtered.append(poi_coor[row])
        else:
            for id in row:
                coor_filtered.append(poi_coor[id])
    return coor_filtered


'''
将 poi 根据轨迹头尾节点以及半径进行过滤
输入的 kdtree 和 poi_coor 中的 poi 坐标单位是米
返回的POI列表是以 经纬度 为坐标单位的
'''
def getPOI_filter_by_radius(trips, poi_coor, kdtree, radius):
    points = []
    for trip in trips:
        x, y = lonlat2meters(trip[0][0], trip[0][1])
        points.append([x, y])
        x, y = lonlat2meters(trip[-1][0], trip[-1][1])
        points.append([x, y])
    id_list = kdtree.query_ball_point(points, radius)
    coor_filtered = []
    for row in id_list:
        # 加判断，如果 row 为空 list，则用 topk 的方式取一次 POI？
        if len(points) == 1:
            x, y = meters2lonlat(poi_coor[row][0], poi_coor[row][1])
            coor_filtered.append([x, y])
        else:
            for id in row:
                x, y = meters2lonlat(poi_coor[id][0], poi_coor[id][1])
                coor_filtered.append([x, y])
    return coor_filtered


def lonlat2meters_poi(poi_coor):
    poi_meters_coor = []
    for poi in poi_coor:
        x, y = lonlat2meters(poi[0], poi[1])
        poi_meters_coor.append([x, y])
    return  poi_meters_coor


def lonlat2meters_coords(coords, use_time=False):
    meters_coor = []
    for coord in coords:
        (x, y), time_stamp, trj_id, flag = (lonlat2meters(coord[0], coord[1]), coord[2], coord[3], coord[4])
        if use_time:
            meters_coor.append([x, y, time_stamp, trj_id, flag])
        else:
            meters_coor.append([x, y, trj_id, flag])
    return meters_coor


def meters2lonlat_list(coord_lst):
    ans = []
    for p in coord_lst:
        x, y = meters2lonlat(p[0], p[1])
        ans.append([x, y])
    return ans


def showPOI_Coor(poi_coor):
    fig = plt.figure(figsize=(20, 10))
    ax = fig.subplots()
    print('poi coord', poi_coor)
    ax.scatter(poi_coor[:, 0], poi_coor[:, 1], c='g', marker='o')
    ax.set_xlabel('lon')  # 画出坐标轴
    ax.set_ylabel('lat')
    plt.show()


if __name__ == "__main__":
    with open("../conf/graph_gen.json") as conf:
        json_data = json.load(conf)
        poi_dir = json_data['poi_dir']
        poi_file_name_lst = json_data['poi_file_name_lst']

        showPOI_Coor(getPOI_Coor(poi_dir, poi_file_name_lst))

# data_dir = '../../hangzhou-POI'
# print(os.listdir(data_dir))
# file_list = os.listdir('../../hangzhou-POI')
# for i in range(len(file_list)):
#     file_list[i] = data_dir + '/' + file_list[i]
#     print(file_list[i])
#
# # ['名称', '大类', '中类', '小类', '地址', '省', '市', '区', 'WGS84_经度', 'WGS84_纬度']
# df1 = pd.read_excel('../../hangzhou-POI/商务住宅.xlsx')  # 读取xlsx中的第一个sheet
# # print('当前文件: ', file_list[4])
# data1 = df1.head(10)  # 读取前10行所有数据
# # data2 = df1.values  # list【】  相当于一个矩阵，以行为单位
# # print(df1.keys())
#
# poi_coor = df1.iloc[:, [-2, -1]].values
# print(poi_coor)
# min_longitude = min(poi_coor[:, 0])
# min_latitude = min(poi_coor[:, 1])
# max_longitude = max(poi_coor[:, 0])
# max_latitude = max(poi_coor[:, 1])
#
# fig = plt.figure(figsize=(20, 10))
# ax = fig.subplots()
#
# ax.scatter(poi_coor[:, 0], poi_coor[:, 1], c='g', marker='o')
# ax.set_xlabel('lon')  # 画出坐标轴
# ax.set_ylabel('lat')
# plt.show()
