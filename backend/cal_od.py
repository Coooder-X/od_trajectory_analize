import os
import pickle
from datetime import datetime

from database.db_funcs import query_od_by_trj_day_and_hour
from data_process.SpatialRegionTools import get_cell_id_center_coord_dict, makeVocab, inregionS, gps2cell
from data_process.spatial_grid_utils import get_region


def inregionS(region, lon, lat):
    return lon >= region.minlon and lon <= region.maxlon and \
           lat >= region.minlat and lat <= region.maxlat


exp_od_pair_set = {(47, 53), (55, 66), (47, 62), (48, 36), (48, 45), (55, 93), (48, 54), (60, 46), (74, 48), (85, 66), (66, 62), (67, 27), (66, 71), (67, 36), (78, 36), (70, 41), (55, 52), (28, 53), (36, 66), (29, 27), (29, 36), (36, 84), (47, 84), (41, 37), (39, 76), (74, 52), (86, 17), (66, 48), (74, 61), (51, 77), (16, 56), (48, 26), (82, 46), (48, 53), (85, 56), (74, 56), (37, 16), (89, 26), (86, 48), (47, 29), (36, 29), (70, 31), (99, 76), (36, 38), (55, 60), (36, 65), (29, 26), (93, 37), (85, 60), (55, 37), (89, 39), (16, 64), (55, 46), (68, 74), (28, 47), (77, 70), (62, 36), (73, 45), (62, 45), (85, 46), (74, 46), (88, 48), (99, 48), (99, 57), (47, 28), (36, 28), (99, 66), (55, 41), (76, 64), (36, 37), (38, 65), (29, 7), (80, 52), (9, 47), (27, 55), (62, 31), (54, 27), (39, 56), (93, 36), (84, 55), (95, 64), (96, 56), (55, 27), (37, 77), (36, 32), (46, 64), (17, 37), (69, 47), (69, 56), (91, 62), (61, 70), (39, 42), (94, 76), (85, 27), (87, 64), (88, 29), (76, 36), (65, 36), (56, 67), (53, 80), (99, 56), (47, 27), (26, 53), (49, 55), (38, 64), (77, 46), (27, 36), (15, 89), (84, 36), (56, 53), (45, 53), (46, 27), (46, 36), (37, 76), (46, 54), (69, 28), (58, 46), (8, 27), (19, 27), (8, 36), (71, 74), (91, 61), (29, 86), (87, 27), (60, 73), (87, 36), (44, 65), (37, 44), (37, 53), (49, 27), (86, 76), (49, 36), (38, 36), (77, 27), (49, 54), (70, 86), (94, 52), (39, 27), (41, 46), (83, 61), (52, 55), (75, 57), (41, 64), (56, 52), (46, 17), (89, 76), (69, 27), (55, 92), (8, 17), (7, 61), (60, 36), (56, 29), (56, 38), (64, 51), (56, 47), (99, 27), (97, 57), (79, 36), (36, 74), (18, 47), (79, 63), (48, 66), (41, 27), (41, 36), (52, 36), (52, 45), (75, 56), (53, 37), (64, 46), (37, 29), (53, 46), (56, 51), (86, 52), (76, 29), (89, 48), (78, 48), (67, 57), (71, 27), (15, 37), (26, 37), (82, 54), (29, 48), (48, 61), (83, 46), (94, 46), (44, 27), (60, 53), (72, 36), (45, 28), (45, 37), (37, 42), (55, 50), (38, 16), (36, 55), (36, 64), (79, 53), (53, 27), (37, 28), (67, 38), (55, 36), (78, 56), (67, 56), (89, 56), (70, 61), (28, 64), (29, 47), (83, 27), (81, 66), (54, 58), (74, 36), (39, 78), (74, 45), (86, 37), (86, 46), (36, 27), (67, 42), (16, 58), (55, 49), (81, 61), (54, 62), (66, 27), (66, 36), (74, 58), (55, 17), (85, 67), (95, 99), (16, 44), (55, 26), (67, 28), (78, 37), (55, 44), (67, 46), (70, 51), (58, 55), (27, 94), (84, 76), (74, 53), (85, 53), (66, 58), (47, 17), (76, 62), (55, 48), (65, 80), (9, 27), (47, 44), (46, 76), (46, 85), (61, 46), (39, 36), (78, 27), (36, 21), (46, 53), (36, 39), (69, 36), (58, 36), (38, 76), (99, 36), (36, 7), (65, 52), (16, 47), (68, 48), (38, 53), (77, 44), (46, 84), (77, 62), (7, 74), (27, 61), (91, 64), (39, 53), (84, 52), (87, 48), (66, 25), (55, 15), (46, 52), (38, 66), (27, 29), (58, 53), (69, 53), (42, 36), (27, 47), (54, 37), (53, 41), (84, 56), (95, 56), (46, 29), (57, 29), (76, 60), (37, 87), (79, 62), (90, 71), (53, 36), (87, 56), (65, 19), (37, 46), (37, 55), (65, 37), (76, 37), (37, 64), (65, 46), (76, 46), (57, 42), (46, 42), (76, 55), (36, 86), (79, 66), (27, 37), (27, 46), (75, 68), (41, 66), (45, 27), (86, 55), (56, 63), (86, 64), (57, 28), (7, 36), (55, 85), (90, 61), (39, 15), (64, 26), (74, 67), (93, 80), (37, 27), (85, 76), (37, 36), (37, 45), (76, 27), (65, 27), (37, 54), (38, 19), (89, 55), (78, 64), (67, 64), (55, 71), (26, 44), (36, 76), (47, 76), (36, 85), (83, 53), (64, 21), (41, 65)}

def get_od_filter_by_day_and_hour(month, start_day, end_day, start_hour, end_hour, region):
    # =============== 旧的实现 ==================
    # res = []
    # start_time = datetime.now()
    # for i in range(start_day, end_day + 1):
    #     data_target_path = "/home/zhengxuan.lin/project/tmp/" + "2020" + str(month).zfill(2) + str(i).zfill(2) + "_trj.pkl"
    #     data_source_path = "/home/zhengxuan.lin/project/" + str(month) + "月/" + str(month).zfill(2) + "月" + str(i).zfill(
    #         2) + "日/2020" + str(month).zfill(2) + str(i).zfill(
    #         2) + "_hz.h5"
    #     if not os.path.exists(data_target_path):
    #         pass
    #     with open(data_target_path, 'rb') as file:
    #         trjs = pickle.loads(file.read())
    #         print(i, trjs[0])
    #     print('读取文件结束，用时: ', (datetime.now() - start_time))
    #     # print(len(od_points), od_points)  # 读取文件结束，用时:  0:00:00.004556
    #     for (idx, trj) in enumerate(trjs):
    #         t = trj[2:]
    #         o, d = t[0], t[-1]
    #         if inregionS(region, o[0], o[1]) and inregionS(region, d[0], d[1]) and \
    #                 start_hour * 3600 <= o[2] <= end_hour * 3600 and\
    #                 start_hour * 3600 <= d[2] <= end_hour * 3600:
    #             res.append([o, d])   # trj[0] 是轨迹的id，trj[1]是日期，trj[2:]是轨迹点序列
    # =============== 基于数据库的实现 ==================
    res = query_od_by_trj_day_and_hour(month, start_day, end_day, start_hour, end_hour, region)
    return res


def encode_od_point(p):
    lon, lat, time = p[0], p[1], p[2]
    return f'{lon}_{lat}_{time}'


def get_od_hot_cell(od_pairs, region, k, lower_bound):
    od_pair_num_dict = {}
    od_cell_set = set()
    hot_od_gps_set = set()
    for od in od_pairs:
        o, d = od[0], od[1]
        o_cell = gps2cell(region, o[0], o[1])
        d_cell = gps2cell(region, d[0], d[1])
        od_cell = (o_cell, d_cell)
        if od_cell not in od_pair_num_dict:
            od_pair_num_dict[od_cell] = 0
        od_pair_num_dict[od_cell] += 1
        if od_pair_num_dict[od_cell] > lower_bound:
            hot_od_gps_set.add(encode_od_point(od[0]))
            hot_od_gps_set.add(encode_od_point(od[1]))
    # print(od_pair_num_dict)

    data = [[od_pair, num] for od_pair, num in od_pair_num_dict.items()]
    data = sorted(data, key=lambda x: x[1], reverse=True)[0:k]
    od_pair_set = set()
    cnt = 0
    od_flow_dict = {}
    for od_pair, num in od_pair_num_dict.items():
        od_flow_dict[od_pair] = num
    for d in data:
        if d[1] > lower_bound:
            cnt += 1
            od_pair_set.add(d[0])
            print(f'od: {d[0]}  流量：{d[1]}')

    with open(f'./od_flow_dict.pkl', 'wb') as f:
        picklestring = pickle.dumps({
            'od_flow_dict': od_flow_dict,
        })
        f.write(picklestring)
        f.close()
    print(f'OD对数量：{cnt}/{len(data)}')
    for od in data:
        o_cell, d_cell = od[0][0], od[0][1]
        od_cell_set.add(o_cell)
        od_cell_set.add(d_cell)
        # print(f'od={o_cell, d_cell}')
    print('使用到的OD区域数', len(od_cell_set))
    print(od_cell_set)
    # for i in range(100):
    #     if i not in od_cell_set:
    #         print(i)
    return data, od_cell_set, od_pair_set, hot_od_gps_set


if __name__ == '__main__':
    od_region = get_region()
    print(len(od_region.centers))
    print(od_region.numx, od_region.numy)
    # print(od_region.maxx - od_region.minx)
    # print(od_region.maxy - od_region.miny)
    # print(od_region.xstep, od_region.ystep)
    # total_od_pairs = get_od_filter_by_day_and_hour(5, 11, 15, 8, 9, od_region)
    # # print(total_od_pairs[0:3])
    # od_pairs, od_cell_set, od_pair_set, hot_od_gps_set = get_od_hot_cell(total_od_pairs, od_region, 1000, 1)
    # print(od_pair_set)
    tmp = get_od_filter_by_day_and_hour(5, 12, 14, 8, 10, od_region)
    print('====> D1的OD对个数：', len(tmp))
    total_od_pairs = get_od_filter_by_day_and_hour(5, 1, 30, 0, 24, od_region)
    # print(total_od_pairs[0:3])
    od_pairs, od_cell_set, od_pair_set, hot_od_gps_set = get_od_hot_cell(total_od_pairs, od_region, 1000, 0)
    print(od_pair_set)