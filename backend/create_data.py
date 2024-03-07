# -*- coding: utf-8 -*-
import threading
from datetime import time

import math
import numpy as np

import pandas as pd

from cal_od import get_od_filter_by_day_and_hour, get_od_hot_cell
from data_process.SpatialRegionTools import gps2cell
from data_process.spatial_grid_utils import get_region

def get_od_flow(od_pairs, od_region, hot_od_cell_set):
    od_pair_num_dict = {}
    for od in od_pairs:
        o, d = od[0], od[1]
        o_cell = gps2cell(od_region, o[0], o[1])
        d_cell = gps2cell(od_region, d[0], d[1])
        #  过滤，只取热门OD对
        if o_cell not in hot_od_cell_set or d_cell not in hot_od_cell_set:
            continue
        od_cell = (o_cell - 1, d_cell - 1)
        if od_cell not in od_pair_num_dict:
            od_pair_num_dict[od_cell] = 0
        od_pair_num_dict[od_cell] += 1
    return od_pair_num_dict


def get_row_col_id(cell_id):
    row_id = cell_id // 10
    col_id = cell_id % 10
    return row_id, col_id


# def get_trj_filter_by_day_and_hour(month, start_day, end_day, start_hour, end_hour, region):
#     res = []
#     start_time = datetime.now()
#     for i in range(start_day, end_day + 1):
#         data_target_path = "/home/zhengxuan.lin/project/tmp/" + "2020" + str(month).zfill(2) + str(i).zfill(2) + "_trj.pkl"
#         data_source_path = "/home/zhengxuan.lin/project/" + str(month) + "月/" + str(month).zfill(2) + "月" + str(i).zfill(
#             2) + "日/2020" + str(month).zfill(2) + str(i).zfill(
#             2) + "_hz.h5"
#         if not os.path.exists(data_target_path):
#             pass
#         with open(data_target_path, 'rb') as file:
#             trjs = pickle.loads(file.read())
#             print(i, trjs[0])
#         print('读取文件结束，用时: ', (datetime.now() - start_time))
#         # print(len(od_points), od_points)  # 读取文件结束，用时:  0:00:00.004556
#         for (idx, trj) in enumerate(trjs):
#             t = trj[2:]
#             o, d = t[0], t[-1]
#             if inregionS(region, o[0], o[1]) and inregionS(region, d[0], d[1]) and \
#                     start_hour * 3600 <= o[2] <= end_hour * 3600 and\
#                     start_hour * 3600 <= d[2] <= end_hour * 3600:
#                 res.append([o, d])   # trj[0] 是轨迹的id，trj[1]是日期，trj[2:]是轨迹点序列
#     return res


def get_od_filter_by_hour(today_od_pairs, start_hour, end_hour):
    res = []
    for od_pairs in today_od_pairs:
        o, d = od_pairs[0], od_pairs[1]
        if start_hour * 3600 <= o[2] <= end_hour * 3600 and\
                start_hour * 3600 <= d[2] <= end_hour * 3600:
            res.append([o, d])   # trj[0] 是轨迹的id，trj[1]是日期，trj[2:]是轨迹点序列
    return res


def create_grid_od(hot_od_cell_set, od_limit):
    data = {
        'dyna_id': [],
        'type': [],
        'time': [],
        'origin_row_id': [],
        'origin_column_id': [],
        'destination_row_id': [],
        'destination_column_id': [],
        'flow': []
    }
    time_slice_dict = {}
    row_num = 10
    col_num = 10
    cell_num = 100
    total_slice = 31 * 24 * 4  # 时间片为15min，五月份总共的时间片数量
    cur_day = 1
    cur_hour = 0
    today_od_pairs = None
    for time_slice in range(total_slice):
        if math.floor(cur_hour + 0.251) - 1 > math.floor(cur_hour):
            next_hour = cur_hour + 1
        else:
            next_hour = cur_hour + 0.25

        # print(cur_day, cur_hour, next_hour)
        if cur_hour == 0:  # 如果是新的一天，则读一次文件，获取当天的OD对数据
            today_od_pairs = get_od_filter_by_day_and_hour(5, cur_day, cur_day, 0, 24, od_region)

        part_od_pairs = get_od_filter_by_hour(today_od_pairs, cur_hour, next_hour)
        od_pair_num_dict = get_od_flow(part_od_pairs, od_region, hot_od_cell_set)
        for cell_id_i in range(cell_num):
            for cell_id_j in range(cell_num):
                od_flow = od_pair_num_dict[(cell_id_i, cell_id_j)] if (cell_id_i, cell_id_j) in od_pair_num_dict else 0
                origin_row_id, origin_column_id = get_row_col_id(cell_id_i)
                destination_row_id, destination_column_id = get_row_col_id(cell_id_j)
                if time_slice not in time_slice_dict:
                    time_slice_dict[time_slice] = {}
                time_slice_dict[time_slice][(cell_id_i, cell_id_j)] = {
                    'dyna_id': time_slice,
                    'type': 'state',
                    'time': None,  # 需要在最后写入文件时改为 NaN
                    'origin_row_id': origin_row_id,
                    'origin_column_id': origin_column_id,
                    'destination_row_id': destination_row_id,
                    'destination_column_id': destination_column_id,
                    'flow': od_flow
                }
        cur_hour = next_hour
        # 如果跨天了
        if math.fabs(24 - next_hour) < 0.001:
            cur_day += 1
            cur_hour = 0
    for cell_id_i in range(cell_num):
        for cell_id_j in range(cell_num):
            for time_slice in range(total_slice):
                if time_slice_dict[time_slice][(cell_id_i, cell_id_j)]['flow'] > 0:
                    print('time:', time_slice, 'grid:', cell_id_i, cell_id_j, 'flow =',
                          time_slice_dict[time_slice][(cell_id_i, cell_id_j)]['flow'])
                # print('time:', time_slice, 'grid:', cell_id_i, cell_id_j, 'flow =', time_slice_dict[time_slice][(cell_id_i, cell_id_j)]['flow'])
                # print(time_slice_dict[time_slice].keys())
                data['dyna_id'].append(time_slice_dict[time_slice][(cell_id_i, cell_id_j)]['dyna_id'])
                data['type'].append(time_slice_dict[time_slice][(cell_id_i, cell_id_j)]['type'])
                data['origin_row_id'].append(time_slice_dict[time_slice][(cell_id_i, cell_id_j)]['origin_row_id'])
                data['origin_column_id'].append(time_slice_dict[time_slice][(cell_id_i, cell_id_j)]['origin_column_id'])
                data['destination_row_id'].append(time_slice_dict[time_slice][(cell_id_i, cell_id_j)]['destination_row_id'])
                data['destination_column_id'].append(time_slice_dict[time_slice][(cell_id_i, cell_id_j)]['destination_column_id'])
                data['flow'].append(time_slice_dict[time_slice][(cell_id_i, cell_id_j)]['flow'])
    data['time'] = np.nan
    df = pd.DataFrame(data)
    print('begin writing file')
    file_name = '/home/zhengxuan.lin/project/Bigscity-LibCity/raw_data/NYC_TOD/NYC_TOD_' + str(od_limit) + '.gridod'
    df.to_csv(file_name, index=True)
    print('finish writing')


if __name__ == '__main__':
    def print_time():  # 无限循环
        print('-------------------->>>>  start')
        while True:  # 获取当前的时间
            current_time = time.ctime(time.time())  # 输出线程的名字和时间
            print('keep live', current_time)  # 休眠10分钟，即600秒 time.sleep(600)
            time.sleep(600)


    thread = threading.Thread(target=print_time)
    thread.start()

    od_limit = 1000

    od_region = get_region()

    total_od_pairs = get_od_filter_by_day_and_hour(5, 1, 3, 0, 24, od_region)
    od_pairs, od_cell_set, od_pair_set, hot_od_gps_set = get_od_hot_cell(total_od_pairs, od_region, od_limit, 0)
    od_cell_set = [i for i in range(1, 101)]
    # print(od_pair_set)
    create_grid_od(od_cell_set, od_limit)
