import os
import pickle
from datetime import datetime

from data_process.SpatialRegionTools import get_cell_id_center_coord_dict, makeVocab, inregionS, gps2cell
from data_process.spatial_grid_utils import get_region


def inregionS(region, lon, lat):
    return lon >= region.minlon and lon <= region.maxlon and \
           lat >= region.minlat and lat <= region.maxlat


def get_od_filter_by_day(month, start_day, end_day, region):
    res = []
    start_time = datetime.now()
    for i in range(start_day, end_day + 1):
        data_target_path = "/home/zhengxuan.lin/project/tmp/" + "2020" + str(month).zfill(2) + str(i).zfill(2) + "_trj.pkl"
        data_source_path = "/home/zhengxuan.lin/project/" + str(month) + "月/" + str(month).zfill(2) + "月" + str(i).zfill(
            2) + "日/2020" + str(month).zfill(2) + str(i).zfill(
            2) + "_hz.h5"
        if not os.path.exists(data_target_path):
            pass
        with open(data_target_path, 'rb') as file:
            trjs = pickle.loads(file.read())
            print(i, trjs[0])
        print('读取文件结束，用时: ', (datetime.now() - start_time))
        # print(len(od_points), od_points)  # 读取文件结束，用时:  0:00:00.004556
        for (idx, trj) in enumerate(trjs):
            t = trj[2:]
            o, d = t[0], t[-1]
            if inregionS(region, o[0], o[1]) and inregionS(region, d[0], d[1]):
                res.append([o, d])   # trj[0] 是轨迹的id，trj[1]是日期，trj[2:]是轨迹点序列
    return res


def get_od_hot_cell(od_pairs, region):
    od_pair_num_dict = {}
    od_cell_set = set()
    for od in od_pairs:
        o, d = od[0], od[1]
        o_cell = gps2cell(region, o[0], o[1])
        d_cell = gps2cell(region, d[0], d[1])
        od_cell = (o_cell, d_cell)
        if od_cell not in od_pair_num_dict:
            od_pair_num_dict[od_cell] = 0
        od_pair_num_dict[od_cell] += 1
    # print(od_pair_num_dict)

    data = [[od_pair, num] for od_pair, num in od_pair_num_dict.items()]
    data = sorted(data, key=lambda x: x[1], reverse=True)[0:1000]
    for od in data:
        o_cell, d_cell = od[0][0], od[0][1]
        od_cell_set.add(o_cell)
        od_cell_set.add(d_cell)
        print(f'od={o_cell, d_cell}')
    print('使用到的OD区域数', len(od_cell_set))
    print(od_cell_set)
    for i in range(100):
        if i not in od_cell_set:
            print(i)
    return data, od_cell_set


if __name__ == '__main__':
    od_region = get_region()
    print(len(od_region.centers))
    print(od_region.numx, od_region.numy)
    # print(od_region.maxx - od_region.minx)
    # print(od_region.maxy - od_region.miny)
    # print(od_region.xstep, od_region.ystep)
    total_od_pairs = get_od_filter_by_day(5, 1, 30, od_region)
    # print(total_od_pairs[0:3])
    od_pairs, od_cell_set = get_od_hot_cell(total_od_pairs, od_region)