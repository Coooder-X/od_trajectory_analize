import json
import math
import pickle
from datetime import datetime

import numpy as np

from data_process.DT_graph_clustering import od_points_filter_by_hour
from data_process.hierarchical_clustering import get_trip_endpoints
from poi_process.read_poi import lonlat2meters_coords
from vis.trajectoryVIS import FileInfo
import os
import h5py

from utils import lonlat2meters


def get_data():
    fileInfo = FileInfo()
    return get_trip_endpoints(fileInfo, 1, False)


def get_hour_od_points():
    start_time = datetime.now()
    #  D:/研究生/chinavis2023/od_trajectory_analize/backend/data/全天OD点经纬度(带轨迹id).pkl
    # /home/linzhengxuan/project/od_trajectory_analize/backend/data/
    with open("/home/linzhengxuan/project/od_trajectory_analize/backend/data/全天OD点经纬度(带轨迹id).pkl", 'rb') as file:
        od_points = pickle.loads(file.read())
    print('读取文件结束，用时: ', (datetime.now() - start_time))
    # print(len(od_points), od_points)  # 读取文件结束，用时:  0:00:00.004556
    res = []
    num_of_points = len(od_points) / 5
    start_id = len(od_points) / 3
    for (idx, od) in enumerate(od_points):
        # if idx % 12 == 0:
        #     res.append(od.tolist())
        if start_id <= idx <= start_id + num_of_points:
            res.append(od.tolist())
    return res


def get_total_od_points():
    start_time = datetime.now()
    with open("/home/linzhengxuan/project/od_trajectory_analize/backend/data/全天OD点经纬度(带轨迹id).pkl", 'rb') as file:
        od_points = pickle.loads(file.read())
    print('读取文件结束，用时: ', (datetime.now() - start_time))
    # print(len(od_points), od_points)  # 读取文件结束，用时:  0:00:00.004556
    res = []
    for (idx, od) in enumerate(od_points):
        res.append(od.tolist())
    return res


def get_od_points_filter_by_hour(start_hour, end_hour):
    od_points = np.asarray(get_total_od_points())
    (part_od_coord_points, index_lst) = od_points_filter_by_hour(od_points, start_hour, end_hour)  # 过滤出所有在该时间段的 od 点
    return {'part_od_points': part_od_coord_points.tolist(), 'index_lst': index_lst[0].tolist()}


# def get_trj_by_od_ids(od_index: list):


# def get_odcluster_trjs(od_idx_in_cluster):
#     """
#     根据一个 od 点 id 数组，得到对应的 轨迹数据
#     :param od_idx_in_cluster: 一个数组，包含了一个簇内的所有 OD 点 id
#     """
#     pass


def get_odpair_space_similarity(cid_lst: list, cid_center_coord_dict, force_nodes):
    """
    计算 OD 对之间的距离（并非相似度，但可以改为相似度）
    :param cid_center_coord_dict: Map<簇id, [lon, lat]> 的映射，存储簇中心点的坐标
    :param force_nodes: 线图的节点，用于构建所有线图边的距离map的key
    :return edge_name_dist_map: 经过计算的 OD 对距离map，形如 map<'12_34-45_78', dist>，key 是用'-'隔开的两个OD对的name
    """
    print('簇个数 =', len(cid_lst))

    # OD对名称到OD对坐标的映射：map<'12_34', [[x0,y0], [x1,y1]]>，键是两个簇id拼接，值是米为单位的坐标
    pair_name_coord_map = {}
    # 以OD对名称之间加‘-’的拼接作为边的key，距离/相似度作为值的map：map<'12_34-58_23', value>
    edge_name_dist_map = {}
    for node in force_nodes:
        od_pair = node['name']
        pair_o, pair_d = list(map(int, od_pair.split('_')))
        lon_meter_o, lat_meter_o = lonlat2meters(cid_center_coord_dict[pair_o][0], cid_center_coord_dict[pair_o][1])
        lon_meter_d, lat_meter_d = lonlat2meters(cid_center_coord_dict[pair_d][0], cid_center_coord_dict[pair_d][1])
        pair_name_coord_map[od_pair] = np.array([[lon_meter_o, lat_meter_o], [lon_meter_d, lat_meter_d]])

    def spatial_dist(od1, od2):
        # 计算两个OD对之间的起点和终点的欧氏距离
        d1 = np.linalg.norm(od1[0] - od2[0])
        d2 = np.linalg.norm(od1[1] - od2[1])
        # 使用指数函数转换为空间相似度
        s1 = d1 + d2

        d1 = np.linalg.norm(od1[0] - od2[1])
        d2 = np.linalg.norm(od1[1] - od2[0])
        s2 = d1 + d2
        return min(s1, s2)

    min_s, max_s = math.inf, -math.inf
    for i in range(len(force_nodes)):
        od_pair1 = force_nodes[i]['name']
        coord_pair1 = pair_name_coord_map[od_pair1]
        for j in range(i+1, len(force_nodes)):
            od_pair2 = force_nodes[j]['name']
            coord_pair2 = pair_name_coord_map[od_pair2]
            dist = spatial_dist(coord_pair1, coord_pair2) / 100
            edge_name_dist_map[f'{od_pair1}-{od_pair2}'] = dist
            # print(f"OD pair {od_pair1}+{od_pair2} have a spatial similarity of {dist:.3f}")
            min_s = min(min_s, dist)
            max_s = max(max_s, dist)

    return edge_name_dist_map


def get_od_points_filter_by_day_and_hour(start_day, end_day, start_hour=0, end_hour=24):
    od_points = np.asarray(get_total_od_points_by_day(start_day, end_day))
    (part_od_coord_points, index_lst) = od_points_filter_by_hour(od_points, start_hour, end_hour)  # 过滤出所有在该时间段的 od 点
    return {'od_points': part_od_coord_points.tolist(), 'index_lst': index_lst[0].tolist()}


def get_total_od_points_by_day(start_day, end_day):
    res = []
    start_time = datetime.now()
    for i in range(start_day, end_day + 1):
        data_target_path = "/tmp/" + "202005" + str(i).zfill(2) + ".pkl"
        data_source_path = "/home/linzhengxuan/project/5月/05月" + str(i).zfill(2) + "日/202005" + str(i).zfill(
            2) + "_hz.h5"
        if not os.path.exists(data_target_path):
            filter_step = 1
            get_odpoints_and_save_as_pkl_file(data_source_path, data_target_path, filter_step)
        with open(data_target_path, 'rb') as file:
            od_points = pickle.loads(file.read())
        print('读取文件结束，用时: ', (datetime.now() - start_time))
        # print(len(od_points), od_points)  # 读取文件结束，用时:  0:00:00.004556
        for (idx, od) in enumerate(od_points):
            res.append(od.tolist())
    return res


def get_odpoints_and_save_as_pkl_file(data_source_path, data_target_path, filter_step, use_cell=False):
    od_points = get_endpoints(data_source_path, filter_step, use_cell)
    start_time = datetime.now()
    with open(data_target_path, 'wb') as f:
        picklestring = pickle.dumps(od_points)
        f.write(picklestring)
    print('写入文件结束，用时: ', (datetime.now() - start_time))


def get_endpoints(data_source_path, filter_step, day, use_cell=False):
    points = []
    trips, lines = get_trips_and_lines(data_source_path, filter_step, use_cell)
    for index, trip in enumerate(trips):
        points.append(trip[0])
        points.append(trip[-1])

    return points


def get_trips_and_lines(data_source_path, filter_step, use_cell=False):
    print(os.getcwd())
    with open("/home/linzhengxuan/project/od_trajectory_analize/backend/data/region.pkl", 'rb') as file:
        region = pickle.loads(file.read())

    # '../make_data/20200101_jianggan.h5'
    with h5py.File(data_source_path, 'r') as f:
        print('轨迹数: ', len(f['trips']))
        trips = []
        lines = []
        for i in range(0, len(f['trips']), filter_step):  # , 1600):
            locations = f['trips'][str(i + 1)]
            trip = []
            line = []
            if region.needTime:
                timestamp = f["timestamps"][str(i + 1)]
                for ((lon, lat), time) in zip(locations, timestamp):
                    if use_cell:  # 将 GPS经纬度表示的轨迹 转换为 网格点经纬度表示
                        cell = gps2cell(region, lon, lat)
                        x, y = cell2coord(region, cell)
                        trip.append([x, y, time])
                    else:
                        trip.append([lon, lat, time])
            else:
                for (lon, lat) in locations:
                    if use_cell:  # 将 GPS经纬度表示的轨迹 转换为 网格点经纬度表示
                        cell = gps2cell(region, lon, lat)
                        x, y = cell2coord(region, cell)
                        trip.append([x, y])
                    else:
                        trip.append([lon, lat])
            # print(trip)
            for j in range(len(trip) - 1):
                line.append([(trip[j][0], trip[j][1]), (trip[j + 1][0], trip[j + 1][1])])
            lines.append(line)
            trip = np.array(trip)
            trips.append(trip)

        return trips, lines

def trj_num_by_hour(date):
    res = []
    for i in range(24):
        res.append(len(get_od_points_filter_by_day_and_hour(date, date, i, i + 1)['od_points']))
    return res

if __name__ == '__main__':
    # print('开始读取OD点')
    # start_time = datetime.now()
    # od_points = np.asarray(get_data())
    # print(od_points[0])
    # print('读取OD点结束，用时: ', (datetime.now() - start_time))
    #
    # start_time = datetime.now()
    # with open("../data/全天OD点经纬度(带轨迹id).pkl", 'wb') as f:
    #     picklestring = pickle.dumps(od_points)
    #     f.write(picklestring)
    # print('写入文件结束，用时: ', (datetime.now() - start_time))

    start_time = datetime.now()
    with open("../data/全天OD点经纬度(带轨迹id).pkl", 'rb') as file:
        od_points = pickle.loads(file.read())
    print('读取文件结束，用时: ', (datetime.now() - start_time))
    print(len(od_points), od_points)  # 读取文件结束，用时:  0:00:00.004556

    # hour_od_pairs = od_points[(14*3600 < od_points[:, 2]) & (od_points[:, 2] < 15*3600)]
    # print(hour_od_pairs, len(hour_od_pairs))
