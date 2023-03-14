import json
import os

import numpy as np
from matplotlib import pyplot as plt

from graph_process.graph_generation import point_compression
from poi_process.read_poi import getPOI_Coor, lonlat2meters_poi, getPOI_filter_by_radius, buildKDTree, \
    meters2lonlat_list
from trip_process.read_trips import getTrips
from utils import lonlat2meters
from vis.trajectoryVIS import randomcolor, FileInfo


def showPOI_in_area(file_info, filter_step, use_cell):
    """
    可视化一个区域内的 POI 和 轨迹端点，前者为绿色，后者为红色。图片保存
    :param file_info: 读取 POI 的文件配置
    :param filter_step: 每隔多少条轨迹，绘制一条轨迹的端点
    :param use_cell:
    """
    trips, lines = getTrips(file_info, filter_step, use_cell)
    poi_coord = getPOI_Coor(file_info.poi_dir, file_info.poi_file_name_lst)
    '''  # 将 POI 根据轨迹端点半径范围过滤
    poi_coord = lonlat2meters_poi(poi_coord) # poi 坐标先转成米为单位，在kdtree中通过半径查找
    kdtree = buildKDTree(poi_coord)
    poi_coord = getPOI_filter_by_radius(trips, poi_coord, kdtree, 500)  # 此处的 poi_coord 是根据轨迹起点、终点过滤后的，坐标单位为经纬度
    '''

    min_longitude = 120.15
    min_latitude = 30.2
    max_longitude = 120.2
    max_latitude = 30.3

    fig = plt.figure(figsize=(20, 10))
    ax = fig.subplots()

    tmp_poi_coord = []
    for coord in poi_coord:
        if coord[0] < min_longitude or coord[0] > max_longitude or coord[1] < min_latitude or coord[1] > max_latitude:
            continue
        tmp_poi_coord.append(coord)
    poi_coord = np.array(tmp_poi_coord)
    ax.scatter(poi_coord[:, 0], poi_coord[:, 1], c='g', marker='o', s=1)

#--------------------------------------------------------------------------------------------
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
    radius = 150
    g, end2poi_dict, poiidx2point_dict = point_compression(trj_point_kdtree, poi_kdtree, endpoints, poi, trj_point_vis,
                                                           end2poi_dict, radius)

    endpoints = meters2lonlat_list(endpoints)
    color_dict = {}
    # ax = fig.subplots()
    print('开始绘图')
    for end_p in end2poi_dict.keys():
        if endpoints[end_p][0] < min_longitude or endpoints[end_p][0] > max_longitude or \
                endpoints[end_p][1] < min_latitude or endpoints[end_p][1] > max_latitude:
            continue
        if end2poi_dict[end_p] in color_dict.keys():
            pass
        else:
            color_dict[end2poi_dict[end_p]] = randomcolor()
        ax.scatter(endpoints[end_p][0], endpoints[end_p][1], c=color_dict[end2poi_dict[end_p]], marker='o', s=1)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # for index, trip in enumerate(trips):
    #     color = 'r'
    #     for j in [0, -1]:
    #         if trip[j][0] < min_longitude or trip[j][0] > max_longitude or trip[j][1] < min_latitude or trip[j][1] > max_latitude:
    #             continue
    #         ax.scatter(trip[j][0], trip[j][1], c=color, marker='o', s=1)

    ax.set_xlabel('lon')  # 画出坐标轴
    ax.set_ylabel('lat')
    plt.savefig(f'../../figs/局部区域POI-根据轨迹端点过滤_{radius}.png', dpi=600, bbox_inches='tight')
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
        print(os.listdir(fileInfo.poi_dir))
        fileInfo.poi_file_name_lst = os.listdir(fileInfo.poi_dir)  # ['商务住宅.xlsx', '风景名胜.xlsx', '购物服务.xlsx', '交通设施服务.xlsx', '医疗保险服务.xlsx', '生活服务.xlsx']
    filter_step = 50
    use_cell = True
    showPOI_in_area(fileInfo, filter_step, False)
