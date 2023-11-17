import numpy as np
from matplotlib import pyplot as plt

from SpatialRegionTools import SpatialRegion, inregionS, gps2cell
from od_pair_process import get_od_points_filter_by_day_and_hour, \
    get_trips_by_ids
from utils import lonlat2meters, meters2lonlat
from vis.trajectoryVIS import randomcolor
from matplotlib import collections as mc

# region_size = 8000
# cellsize = region_size / 10
cellsizeX = 9284.045532159507 / 10
cellsizeY = 8765 / 10
mid_lon, mid_lat = 120.29986, 30.41829
cityname = "hangzhou"
timecellsize = 120
interesttime = 1100.0
nointeresttime = 3300.0
needTime = 0
delta = 1.0
timefuzzysize = 1
timestart = 0
use_grid = 1
has_label = 0


def get_lon_lat_scope(mid_lon, mid_lat, region_size):
    mid_lon, mid_lat = lonlat2meters(mid_lon, mid_lat)
    min_lon = mid_lon - region_size / 2
    min_lat = mid_lat - region_size / 2
    max_lon = min_lon + region_size
    max_lat = min_lat + region_size
    min_lon, min_lat = meters2lonlat(min_lon, min_lat)
    max_lon, max_lat = meters2lonlat(max_lon, max_lat)
    return min_lon, min_lat, max_lon, max_lat


def get_trj_filter_by_region(trj_list, region):
    """
    传入轨迹数据列表，过滤出 “在当前研究区域内的” 轨迹，并返回新的列表。
    注意：“在当前研究区域内的” 的定义有两种：1、轨迹的所有采样点都在区域内   2、只考虑轨迹的头尾在区域内
    这里的实现是基于 第二种定义！
    """
    result = []
    for trj in trj_list:
        head_lon, head_lat = trj[0][0], trj[0][1]
        tail_lon, tail_lat = trj[-1][0], trj[-1][1]
        if inregionS(region, head_lon, head_lat) and inregionS(region, tail_lon, tail_lat):
            result.append(trj)
    return result


def get_od_points_filter_by_region(region, part_od_points, index_lst):
    filtered_od = []
    filtered_idx = []
    for i in range(len(part_od_points)):
        point = part_od_points[i]
        lon, lat = point[0], point[1]
        if inregionS(region, lon, lat):
            filtered_idx.append(index_lst[i])
            filtered_od.append(point)
    return filtered_od, filtered_idx


def divide_od_into_grid(region, part_od_points, index_lst):
    point_cluster_dict = {}
    cluster_point_dict = {}
    for i in range(len(part_od_points)):
        point = part_od_points[i]
        lon, lat = point[0], point[1]
        grid_id = gps2cell(region, lon, lat)
        # print(f'({lon}, {lat})  grid_id = {grid_id}')
        point_cluster_dict[index_lst[i]] = grid_id
        if grid_id not in cluster_point_dict:
            cluster_point_dict[grid_id] = []
        cluster_point_dict[grid_id].append(index_lst[i])

    return point_cluster_dict, cluster_point_dict


def show_trips(trips):
    lines = []
    for trip in trips:
        # trip = np.asarray(trip)
        line = []
        for j in range(len(trip) - 1):
            line.append([(trip[j][0], trip[j][1]), (trip[j + 1][0], trip[j + 1][1])])
        # line = np.asarray(line)
        lines.append(line)
    # trips = np.asarray(trips)
    # lines = np.asarray(lines)

    print('可视化轨迹数量：', len(trips))
    colors = [randomcolor() for i in range(len(trips))]

    fig = plt.figure(figsize=(20, 10))
    ax = fig.subplots()
    for index, line in enumerate(lines):
        color = colors[index]
        lc = mc.LineCollection(line, colors=color, linewidths=2)
        ax.add_collection(lc)
    for index, trip in enumerate(trips):
        trip = np.asarray(trip)
        color = colors[index]
        ax.scatter(trip[:, 0], trip[:, 1], s=1, c=color, marker='o')

    ax.set_xlabel('lon')  # 画出坐标轴
    ax.set_ylabel('lat')
    # plt.show()
    plt.savefig('./spatial_grid_trj.png')


def get_region():
    # min_lon, min_lat, max_lon, max_lat = get_lon_lat_scope(mid_lon, mid_lat, region_size)
    min_lon, min_lat, max_lon, max_lat = 120.1088, 30.2335, 120.1922, 30.3015

    region = SpatialRegion(cityname,
                           min_lon, min_lat,  # 整个hz
                           max_lon, max_lat,  # 整个hz
                           0, 86400,  # 时间范围,一天最大86400(以0点为相对值)
                           cellsizeX, cellsizeY,
                           timecellsize,  # 时间步
                           1,  # minfreq 最小击中次数
                           40_0000,  # maxvocab_size
                           30,  # k
                           4,  # vocab_start 词汇表基准值
                           interesttime,  # 时间阈值
                           nointeresttime,
                           delta,
                           needTime,
                           2, 4000,
                           timefuzzysize, timestart,
                           hulls=None, use_grid=use_grid, has_label=has_label)

    return region


def test():
    region = get_region()

    od_points = get_od_points_filter_by_day_and_hour(5, 1, 2, 8, 10)['od_points']
    trj_ids = []
    trjId_od_dict = {}
    for point in od_points:
        # print('point ===', point)
        point[3] = int(point[3])
        if point[3] not in trjId_od_dict:
            trjId_od_dict[point[3]] = []
        if inregionS(region, point[0], point[1]):
            trjId_od_dict[point[3]].append(point)

    for trj_id in trjId_od_dict:
        if len(trjId_od_dict[trj_id]) >= 2:
            trj_ids.append(trj_id)

    print('len of trjId_od_dict ======>', len(trjId_od_dict.keys()))
    print('trip idx =======>', len(trj_ids), trj_ids)
    gps_trips = get_trips_by_ids(trj_ids=trj_ids, month=5, start_day=1, end_day=2)
    show_trips(gps_trips)


if __name__ == '__main__':
    pass
    # min_lon, min_lat, max_lon, max_lat = get_lon_lat_scope(mid_lon, mid_lat, region_size)
    #
    # region = SpatialRegion(cityname,
    #                            min_lon, min_lat, # 整个hz
    #                            max_lon, max_lat, # 整个hz
    #                            0, 86400,  # 时间范围,一天最大86400(以0点为相对值)
    #                            cellsize, cellsize,
    #                            timecellsize,  # 时间步
    #                            1,  # minfreq 最小击中次数
    #                            40_0000,  # maxvocab_size
    #                            30,  # k
    #                            4,  # vocab_start 词汇表基准值
    #                            interesttime,  # 时间阈值
    #                            nointeresttime,
    #                            delta,
    #                            needTime,
    #                            2, 4000,
    #                            timefuzzysize, timestart,
    #                            hulls=None, use_grid=use_grid, has_label=has_label)
    #
    # od_points = get_od_points_filter_by_day_and_hour(5, 1, 2, 8, 10)
    # trj_ids = []
    # trjId_od_dict = {}
    # for point in od_points:
    #     if point[3] not in trjId_od_dict:
    #         trjId_od_dict[point[3]] = []
    #     if inregionS(region, point[0], point[1]):
    #         trjId_od_dict[point[3]].append(point)
    #
    # for trj_id in trjId_od_dict:
    #     if len(trjId_od_dict[trj_id]) == 2:
    #         trj_ids.append(trj_id)
    #
    # print('len of trjId_od_dict ======>', len(trjId_od_dict.keys()))
    # print('trip idx =======>', trj_ids)
    # gps_trips = get_trips_by_ids(trj_ids=trj_ids, month=5, start_day=1, end_day=2)
    # show_trips(gps_trips)

