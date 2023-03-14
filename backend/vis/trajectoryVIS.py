import json
import os
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy import spatial
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import h5py
import random
import numpy as np
import pickle
from data_process.SpatialRegionTools import gps2vocab, gps2cell, cell2coord

import warnings

from poi_process.read_poi import getPOI_Coor, buildKDTree, getPOI_filter_by_topk, getPOI_filter_by_radius, lonlat2meters_poi
from trip_process.read_trips import getTrips

warnings.simplefilter(action='ignore', category=FutureWarning)


class FileInfo:
    def __init__(self):
        self.trj_data_path = '../../../5月/'
        self.trj_data_date = '05月01日'
        self.trj_file_name = '20200501_hz.h5'
        self.poi_dir = '../../hangzhou-POI'
        self.poi_file_name_lst = ['商务住宅.xlsx', '风景名胜.xlsx']


def randomcolor():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color


colorsOfLabel = [
    "#8ECFC9", "#2878b5", "#F27970", "#A1A9D0", "#C4A5DE", "#63b2ee", "#9192ab", "#3b6291", "#bf7334", "#3b8ba1",
    "#ffbd66",
    "#FFBE7A", "#9ac9db", "#BB9727", "#F0988C", "#F6CAE5", "#76da91", "#7898e1", "#943c39", "#3f6899", "#c97937",
    "#f74d4d",
    "#FA7F6F", "#f8ac8c", "#54B345", "#B883D4", "#96CCCB", "#f8cb7f", "#efa666", "#779043", "#9c403d", "#002c53",
    "#2455a4",
    "#82B0D2", "#c82423", "#32B897", "#9E9E9E", "#8983BF", "#f89588", "#eddd86", "#624c7c", "#7d9847", "#ffa510",
    "#41b7ac",
    "#BEB8DC", "#ff8884", "#05B9E2", "#CFEAF1", "#C76DA2", "#7cd6cf", "#9987ce", "#388498", "#675083", "#0c84c6",
    "#E7DAD2",
    "#63b2ee", "#76da91"
]


def showTrips(fileInfo, filter_step, use_cell=False):
    trips, lines = getTrips(fileInfo, filter_step, use_cell)
    print('可视化轨迹数量：', len(trips))
    colors = [randomcolor() for i in range(len(trips))]

    fig = plt.figure(figsize=(20, 10))
    ax = fig.subplots()
    for index, line in enumerate(lines):
        color = colors[index]
        lc = mc.LineCollection(line, colors=color, linewidths=2)
        ax.add_collection(lc)
    for index, trip in enumerate(trips):
        color = colors[index]
        ax.scatter(trip[:, 0], trip[:, 1], s=1, c=color, marker='o')

    ax.set_xlabel('lon')  # 画出坐标轴
    ax.set_ylabel('lat')
    plt.show()


def showPOI_Trips(fileInfo, filter_step, use_cell):
    trips, lines = getTrips(fileInfo, filter_step, use_cell)
    poi_coor = getPOI_Coor(fileInfo.poi_dir, fileInfo.poi_file_name_lst)
    poi_coor = lonlat2meters_poi(poi_coor) # poi 坐标先转成米为单位，在kdtree中通过半径查找
    kdtree = buildKDTree(poi_coor)
    poi_coor = getPOI_filter_by_radius(trips, poi_coor, kdtree, 500)  # 此处的 poi_coor 是根据轨迹起点、终点过滤后的，坐标单位为经纬度

    min_longitude = float('inf')
    min_latitude = float('inf')
    max_longitude = float('-inf')
    max_latitude = float('-inf')

    print('可视化轨迹数量：', len(trips))


    fig = plt.figure(figsize=(20, 10))
    ax = fig.subplots()

    # 画POI
    for index, trip in enumerate(trips):
        min_longitude = min(min_longitude, min(trip[:, 0]))
        min_latitude = min(min_latitude, min(trip[:, 1]))
        max_longitude = max(max_longitude, max(trip[:, 0]))
        max_latitude = max(max_longitude, max(trip[:, 1]))
    tmp_poi_coor = []
    for coor in poi_coor:
        if coor[0] < min_longitude or coor[0] > max_longitude or coor[1] < min_latitude or coor[1] > max_latitude:
            continue
        tmp_poi_coor.append(coor)
    poi_coor = np.array(tmp_poi_coor)
    # ax.scatter(poi_coor[:, 0], poi_coor[:, 1], c='g', marker='o', s=1)

    # 画轨迹
    colors = [randomcolor() for i in range(len(trips))]
    # for index, line in enumerate(lines):
    #     color = colors[index]
    #     lc = mc.LineCollection(line, colors=color, linewidths=1)
    #     ax.add_collection(lc)
    for index, trip in enumerate(trips):
        color = colors[index]
        ax.scatter(trip[0][0], trip[0][1], c=color, marker='o', s=2)
        ax.scatter(trip[-1][0], trip[-1][1], c=color, marker='o', s=2)

    ax.set_xlabel('lon')  # 画出坐标轴
    ax.set_ylabel('lat')
    plt.savefig('../../figs/trj_and_poi.png', dpi=600, bbox_inches='tight')
    plt.show()


def draw_points(points, color):
    fig = plt.figure(figsize=(20, 10))
    ax = fig.subplots()
    for p in points:
        ax.scatter(p[0], p[1], c=color if color is not None else randomcolor(), marker='o', s=4)

    ax.set_xlabel('lon')  # 画出坐标轴
    ax.set_ylabel('lat')
    plt.show()


if __name__ == "__main__":
    fileInfo = FileInfo()
    with open("../conf/graph_gen.json") as conf:
        # 读取配置文件
        json_data = json.load(conf)
        fileInfo.trj_data_path = json_data['trj_data_path']
        fileInfo.trj_data_date = json_data['trj_data_date']
        fileInfo.trj_file_name = json_data['trj_file_name']
        fileInfo.poi_dir = json_data['poi_dir']
        fileInfo.poi_file_name_lst = json_data['poi_file_name_lst']
    filter_step = 50
    use_cell = True
    # showTrips(fileInfo, filter_step, use_cell)
    showPOI_Trips(fileInfo, filter_step, False) # poi暂时没有网格化，因此同时可视化轨迹和poi时，轨迹也不能网格化，否则坐标不同，报错

    # # print(os.listdir('../../5月/05月01日'))
    # data_path = '../../5月/'
    # data_date = '05月01日'
    # file_name = '20200501_hz.h5'
    # file_path = data_path + data_date + '/' + file_name
    #
    # with open("../data/region.pkl", 'rb') as file:
    #     region = pickle.loads(file.read())
    #
    # fig = plt.figure(figsize=(20, 10))
    # ax = fig.subplots()
    # filter_step = 500
    #
    # # '../make_data/20200101_jianggan.h5'
    # with h5py.File(file_path, 'r') as f:
    #     with h5py.File("../data/hangzhou-vocab-dist-cell250.h5") as kf:
    #         print('轨迹数: ', len(f['trips']))
    #         for i in range(0, len(f['trips']), filter_step):  # , 1600):
    #             locations = f['trips'][str(i + 1)]
    #             trip = []
    #             lines = []
    #             for (lon, lat) in locations:
    #                 trip.append([lon, lat])
    #             seq = np.array(trip2seq(region, trip))
    #             # 将 GPS经纬度表示的轨迹 转换为 网格点经纬度表示
    #             for idx, (lon, lat) in enumerate(trip):
    #                 cell = gps2cell(region, lon, lat)
    #                 x, y = cell2coord(region, cell)
    #                 trip[idx] = [x, y]
    #             # print(trip)
    #             for j in range(len(trip) - 1):
    #                 lines.append([(trip[j][0], trip[j][1]), (trip[j + 1][0], trip[j + 1][1])])
    #             trip = np.array(trip)
    #
    #             color = randomcolor()
    #             lc = mc.LineCollection(lines, colors=color, linewidths=2)
    #             ax.add_collection(lc)
    #             ax.scatter(trip[:, 0], trip[:, 1], c=color, marker='o')
    #
    #         ax.set_xlabel('lon')  # 画出坐标轴
    #         ax.set_ylabel('lat')
    #         handles, labels = ax.get_legend_handles_labels()
    #         handle_list, label_list = [], []
    #         for handle, label in zip(handles, labels):
    #             if label not in label_list:
    #                 handle_list.append(handle)
    #                 label_list.append(label)
    #         plt.legend(handle_list, label_list)
    #         plt.show()
