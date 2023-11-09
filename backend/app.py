import json
import pickle
import random
from datetime import datetime

import numpy as np
from flask import Flask, request
from flask_caching import Cache
from flask_cors import CORS
# from flask_script import Manager
import _thread
import utils
import os

from data_process.SpatialRegionTools import get_cell_id_center_coord_dict
from data_process.spatial_grid_utils import test, get_region, divide_od_into_grid, get_od_points_filter_by_region
from graph_process.Graph import get_adj_matrix, get_feature_list, get_degree_by_node_name, get_dag_from_community
from gcc.graph_convolutional_clustering.gcc.run import run, draw_cluster_in_trj_view
# from model.t2vec import args
from t2vec import args
from t2vec_graph import get_feature_and_trips, run_model2, get_cluster_by_trj_feature
from poi_process.new_read_poi import get_poi_type_filter_by_radius, config_dict, getPOI_Coor, buildKDTree, \
    meters2lonlat_list, lonlat2meters_poi, get_poi_info_lst_by_points
from data_process.od_pair_process import get_odpair_space_similarity, get_trj_ids_by_force_node, get_trips_by_ids, \
    get_trip_detail_by_id
from graph_cluster_test.sa_cluster import update_graph_with_attr, get_cluster
from data_process.OD_area_graph import build_od_graph, get_line_graph_by_selected_cluster, get_cluster_center_coord, \
    fuse_fake_edge_into_linegraph  # , aggregate_single_points
from data_process import od_pair_process
from data_process.DT_graph_clustering import delaunay_clustering, cluster_filter_by_hour, draw_DT_clusters
import os.path
import scipy.io as sio
import scipy.sparse as sp

app = Flask(__name__)
# manager = Manager(app)
CORS(app, resources=r'/*')

#  python -m flask run

# out_adj_dict = {}
line_graph = None
app.config["CACHE_TYPE"] = "simple"
cache = Cache(app)

@app.route('/')
@cache.cached(timeout=0)
def hello_world():  # put application's code here
    # start_time = datetime.now()
    # with open("/home/zhengxuan.lin/project/od_trajectory_analize/backend/data/POI映射关系.pkl", 'rb') as f:
    #     obj = pickle.loads(f.read())
    #     total_poi_coor = obj['total_poi_coor']
    #     file_id_poi_id_dict = obj['file_id_poi_id_dict']
    #     poi_id_file_id_dict = obj['poi_id_file_id_dict']
    #     kdtree = obj['kdtree']
    #     cache.set('file_id_poi_id_dict', file_id_poi_id_dict)
    #     cache.set('poi_id_file_id_dict', poi_id_file_id_dict)
    #     cache.set('total_poi_coor', total_poi_coor)
    #     cache.set('kdtree', kdtree)
    #     print('total_poi_coor', kdtree)
    #     print('读取POI文件结束，用时: ', (datetime.now() - start_time))
    # start_time = datetime.now()
    # total_poi_coor, file_id_poi_id_dict, poi_id_file_id_dict = getPOI_Coor(config_dict['poi_dir'])
    # total_poi_coor = lonlat2meters_poi(total_poi_coor)
    # kdtree = buildKDTree(total_poi_coor)
    # with open("/home/zhengxuan.lin/project/od_trajectory_analize/backend/data/POI映射关系.pkl", 'wb') as f:
    #     picklestring = pickle.dumps({
    #         'total_poi_coor': total_poi_coor,
    #         'file_id_poi_id_dict': file_id_poi_id_dict,
    #         'poi_id_file_id_dict': poi_id_file_id_dict,
    #         'kdtree': kdtree,
    #     })
    #     f.write(picklestring)
    # print('写入文件结束，用时: ', (datetime.now() - start_time))
    return 'Hello World!'


@app.route('/getTotalODPoints', methods=['get', 'post'])
def get_total_od_points():
    # return json.dumps(od_pair_process.get_hour_od_points())
    return json.dumps(od_pair_process.get_total_od_points())


@app.route('/getODPointsFilterByHour', methods=['get', 'post'])
def get_od_points_filter_by_hour():
    start_hour, end_hour = int(request.args['startHour']), int(request.args['endHour'])
    print(start_hour, end_hour)
    # return json.dumps({'od_points':od_pair_process.get_hour_od_points()})
    return json.dumps(od_pair_process.get_od_points_filter_by_hour(start_hour, end_hour))


@app.route('/getODPointsFilterByDayAndHour', methods=['get', 'post'])
def get_od_points_filter_by_day_and_hour():
    month = request.args.get('month', 5, type=int)
    start_day, end_day, start_hour, end_hour = request.args.get('startDay', type=int), \
                                               request.args.get('endDay', type=int), \
                                               request.args.get('startHour', 0, type=int), \
                                               request.args.get('endHour', 24, type=int)
    print(month, start_day, end_day, start_hour, end_hour)
    # return json.dumps({'od_points':od_pair_process.get_hour_od_points()})
    return json.dumps({month: od_pair_process.get_od_points_filter_by_day_and_hour(month, start_day, end_day, start_hour, end_hour)})


@app.route('/getTrjNumByDayAndHour', methods=['get'])
def get_trj_num_by_day_and_hour():
    month = request.args.get('month', 5, type=int)
    start_day, end_day, start_hour, end_hour = request.args.get('startDay', type=int), \
                                               request.args.get('endDay', type=int), \
                                               request.args.get('startHour', 0, type=int), \
                                               request.args.get('endHour', 24, type=int)
    print(start_day, end_day, start_hour, end_hour)
    trj_num = len(od_pair_process.get_trj_num_filter_by_day_and_hour(month, start_day, end_day, start_hour, end_hour)['trips'])
    print(trj_num)
    return json.dumps({'trj_num': trj_num})


@app.route('/getTrjTotalNumByDay', methods=['get'])
def get_total_trj_num_by_day():
    month = request.args.get('month', 5, type=int)
    start_day, end_day = request.args.get('startDay', 1, type=int), request.args.get('endDay', 31, type=int)
    print(start_day, end_day)
    nums = []
    for d in range(start_day, end_day + 1):
        num = len(od_pair_process.get_trj_num_filter_by_day(month, d, d))
        nums.append(num)
    days = [d for d in range(start_day, end_day + 1)]
    nums_dict = dict(zip(days, nums))
    res = {month: nums_dict}
    return json.dumps(res)


@app.route('/getTrjTotalNumByMonth', methods=['get'])
def get_total_trj_num_by_Month():
    month = request.args.get('month', 5, type=int)
    data_path = "/home/zhengxuan.lin/project/tmp/" + str(month).zfill(2) + "trj_num_by_month.txt"
    if not os.path.exists(data_path):
        with open(data_path, "w") as f:
            res = []
            for d in range(31):
                num = len(od_pair_process.get_trj_num_filter_by_day(month, d + 1, d + 1))
                res.append(num)
            for r in res:
                f.write(str(r))
                f.write(' ')
    else:
        res = []
        with open(data_path, 'r') as f:
            line = f.readline()
            line = list(line.strip().split(' '))
            for i in line:
                res.append(int(i))
    return json.dumps(res)


@app.route('/getTrjNumByHour', methods=['get'])
def get_trj_num_by_hour():
    month = request.args.get('month', 5, type=int)
    start_day, end_day = request.args.get('startDay', 1, type=int), request.args.get('endDay', 31, type=int)
    print(start_day, end_day)
    res = od_pair_process.trj_num_by_hour(month, start_day, end_day)
    return json.dumps({month: {'nums': res}})


@app.route('/getTrjNumByOd', methods=['post'])
def get_trj_num_by_od():
    data = request.get_json(silent=True)
    month = data['month']
    start_day, end_day, num = int(data['startDay']), int(data['endDay']), int(data['num'])
    start_day = start_day if start_day - num <= 0 else end_day - num + 1
    src_id_list, tgt_id_list = data['src_id_list'], data['tgt_id_list']
    total_od_points = od_pair_process.get_od_points_filter_by_day_and_hour(month, start_day, end_day, 0, 24)['od_points']
    res = []
    print('start_day, end_day', start_day, end_day + 1)
    for d in range(start_day, end_day):
        num = []
        for h in range(24):
            count = 0
            for src_id in src_id_list:
                for tgt_id in tgt_id_list:
                    # if src_id + 1 == tgt_id:
                    #     count += 1
                    src = total_od_points[src_id]
                    tgt = total_od_points[tgt_id]
                    if src[4] == 0 and tgt[4] == 1 and src[3] == tgt[3]:
                        print('same', d, src[5], h)
                        print(src)
                        print(tgt)
                    if src[4] == 0 and tgt[4] == 1 and src[3] == tgt[3] and src[5] == d and tgt[5] == d and h * 3600 <= \
                            src[2] <= (h + 1) * 3600:
                        count = count + 1
                        # print('same', d, src[5], h)
                        # print(src)
                        # print(tgt)
            num.append(count)
        res.append(num)
    return res


@app.route('/getTripsById', methods=['post'])
def get_trips_by_id():
    data = request.get_json(silent=True)
    month = int(data['month'])
    start_day, end_day, start_hour, end_hour = [int(data['startDay']),
                                                int(data['endDay']),
                                                int(data['startHour']),
                                                int(data['endHour'])]
    trj_ids = data['trjIdList']
    print(start_day, end_day, start_hour, end_hour, trj_ids)
    total_trips = od_pair_process.get_trj_num_filter_by_day_and_hour(month, start_day, end_day, start_hour, end_hour)['trips']

    tid_trip_dict = {}
    for tid in trj_ids:
        tid_trip_dict[tid] = total_trips[tid][3:]

    return json.dumps(
        {
            'tid_trip_dict': tid_trip_dict,
        }
    )


@app.route('/calSpeed', methods=['get'])
def calSpeed():
    month = request.args.get('month', 5, type=int)
    start_day, end_day, start_hour, end_hour = request.args.get('startDay', type=int), \
                                               request.args.get('endDay', type=int), \
                                               request.args.get('startHour', 0, type=int), \
                                               request.args.get('endHour', 24, type=int)
    trj_id = request.args.get('trjId', 0, type=int)
    print(start_day, end_day, start_hour, end_hour, trj_id)
    total_trips = od_pair_process.get_trj_num_filter_by_day_and_hour(month, start_day, end_day, start_hour, end_hour)['trips']
    trips = total_trips[trj_id]
    mean_speed = calTwoPointSpeed(trips[2], trips[len(trips) - 1])
    max_speed = -1
    min_speed = 999999
    speeds = []
    trips_len = len(trips) - 2
    for i in range(3, len(trips)):
        tmp_speed = calTwoPointSpeed(trips[i - 1], trips[i])
        speeds.append(judgeSpeedLevel(tmp_speed))
        min_speed = min(min_speed, tmp_speed)
        max_speed = max(max_speed, tmp_speed)

    mean_speed_level = judgeSpeedLevel(mean_speed)
    max_speed_level = judgeSpeedLevel(max_speed)
    min_speed_level = judgeSpeedLevel(min_speed)
    return json.dumps(
        {
            'trjId': trj_id,
            'startDay': start_day,
            'endDay': end_day,
            'meanSpeed': mean_speed,
            'meanSpeedLevel': mean_speed_level,
            'maxSpeed': max_speed,
            'maxSpeedLevel': max_speed_level,
            'minSpeed': min_speed,
            'minSpeedLevel': min_speed_level,
            'speeds': speeds,
            'trips_len': trips_len
        }
    )


def judgeSpeedLevel(speed):
    if speed <= 20:
        return 1
    elif speed < 40:
        return 2
    elif speed < 60:
        return 3
    else:
        return 4


def calTwoPointSpeed(p0, p1):
    xo, yo = utils.lonlat2meters(p0[0], p0[1])
    xd, yd = utils.lonlat2meters(p1[0], p1[1])
    time = p1[2] - p0[2]
    return utils.cal_meter_dist((xo, yo), (xd, yd)) * 3.6 / time


@app.route('/getGridResult', methods=['get', 'post'])
@cache.cached(timeout=0)
def get_grid_result():
    region = cache.get('region')
    month = int(request.args['month'])
    start_day, end_day = int(request.args['startDay']), int(request.args['endDay'])
    start_hour, end_hour = int(request.args['startHour']), int(request.args['endHour'])

    start_time = datetime.now()
    res = od_pair_process.get_od_points_filter_by_day_and_hour(month, start_day, end_day)
    print(f'start {start_day} end {end_day}')
    od_points = np.array(res['od_points'])
    cache.set('total_od_points', od_points)
    total_od_coord_points = od_points[:, 0:2]  # 并去掉时间戳留下经纬度坐标
    print('读取OD点结束，用时: ', (datetime.now() - start_time))
    res = od_pair_process.get_od_points_filter_by_day_and_hour(month, start_day, end_day, start_hour, end_hour)
    index_lst = res['index_lst']
    part_od_points = res['od_points']

    part_od_points, index_lst = get_od_points_filter_by_region(region, part_od_points, index_lst)
    point_cluster_dict, cluster_point_dict = divide_od_into_grid(region, part_od_points, index_lst)
    out_adj_table, in_adj_table = build_od_graph(point_cluster_dict, od_points, index_lst)

    return json.dumps({
        'index_lst': index_lst,  # 当前小时时间段内的部分 OD 点索引
        'point_cluster_dict': point_cluster_dict,  # 全量的
        'cluster_point_dict': cluster_point_dict,  # 全量的
        'part_cluster_point_dict': cluster_point_dict,  # 当前小时内部分的映射关系，保证每个簇内的点都在当前小时段内
        'part_od_points': part_od_points,  # 当前小时段内部分的 OD 点
        'json_adj_table': {},
        'json_nodes': {},
        'out_adj_table': out_adj_table,  # 当前小时段内过滤处的出边邻接表
        'in_adj_table': in_adj_table,  # 当前小时段内过滤处的入边邻接表
    })


@app.route('/getClusteringResult', methods=['get', 'post'])
@cache.cached(timeout=0)
def get_cluster_result():
    month = int(request.args['month'])
    k, theta = int(request.args['k']), int(request.args['theta'])
    print(f'k={k}, theta:{theta}')
    start_day, end_day = int(request.args['startDay']), int(request.args['endDay'])
    start_hour, end_hour = int(request.args['startHour']), int(request.args['endHour'])

    start_time = datetime.now()
    # od_points = np.asarray(od_pair_process.get_total_od_points())    # get_total_od_points
    res = od_pair_process.get_od_points_filter_by_day_and_hour(month, start_day, end_day)
    print(f'start {start_day} end {end_day}')
    od_points = np.array(res['od_points'])
    cache.set('total_od_points', od_points)

    total_od_coord_points = od_points[:, 0:2]  # 并去掉时间戳留下经纬度坐标
    print('读取OD点结束，用时: ', (datetime.now() - start_time))
    print('pos nums', len(od_points), '\n开始聚类')
    start_time = datetime.now()
    point_cluster_dict, cluster_point_dict = delaunay_clustering(k=k, theta=theta, od_points=total_od_coord_points)
    print('结束聚类，用时: ', (datetime.now() - start_time))

    # res = od_pair_process.get_od_points_filter_by_hour(start_hour, end_hour)
    res = od_pair_process.get_od_points_filter_by_day_and_hour(month, start_day, end_day, start_hour, end_hour)
    index_lst = res['index_lst']
    part_od_points = res['od_points']
    new_point_cluster_dict, new_cluster_point_dict = cluster_filter_by_hour(index_lst, point_cluster_dict)
    print('过滤后的点数：', len(index_lst))
    print('过滤后的的簇数：', len(new_cluster_point_dict.keys()))
    out_adj_table, in_adj_table = build_od_graph(new_point_cluster_dict, od_points, index_lst)
    # print(out_adj_table)
    # draw_DT_clusters(new_cluster_point_dict, total_od_coord_points, k, theta, start_hour, end_hour, set(index_lst))

    total_cluster_point_dict = {}
    for key in cluster_point_dict:
        total_cluster_point_dict[key] = list(cluster_point_dict[key])
    return json.dumps({
        'index_lst': index_lst,  # 当前小时时间段内的部分 OD 点索引
        'point_cluster_dict': point_cluster_dict,  # 全量的
        'cluster_point_dict': total_cluster_point_dict,  # 全量的
        'part_cluster_point_dict': new_cluster_point_dict,  # 当前小时内部分的映射关系，保证每个簇内的点都在当前小时段内
        'part_od_points': part_od_points,  # 当前小时段内部分的 OD 点
        # 'json_adj_table': json_adj_table,
        # 'json_nodes': json_nodes,
        'out_adj_table': out_adj_table,  # 当前小时段内过滤处的出边邻接表
        'in_adj_table': in_adj_table,  # 当前小时段内过滤处的入边邻接表
    })


# @app.route('/getClusterCenter', methods=['post'])
# def get_cluster_center_coords():
#     print('获取簇中心坐标：get_cluster_center_coords')
#     data = request.get_json(silent=True)
#     cluster_point_dict = data['cluster_point_dict']
#     selected_cluster_idxs = data['selected_cluster_idxs']
#
#     tmp = {}
#     for key in cluster_point_dict:
#         tmp[int(key)] = cluster_point_dict[key]
#     cluster_point_dict = tmp
#
#     cid_center_coord_dict = get_cluster_center_coord(cluster_point_dict, selected_cluster_idxs)
#     cache.set('cid_center_coord_dict', cid_center_coord_dict)
#
#     get_odpair_space_similarity(selected_cluster_idxs, cache)
#
#     return json.dumps({
#         'cid_center_coord_dict': cid_center_coord_dict
#     })


# @app.route('/getCommunityCluster', methods=['post'])
# def get_community_cluster():
#     line_graph = cache.get('line_graph')
#     # line_graph = update_graph_with_attr(line_graph)
#     print(line_graph)
#     # 对线图进行图聚类，得到社区发现
#     point_cluster_dict, cluster_point_dict = get_cluster(line_graph, 5)
#     print('point_cluster_dict', point_cluster_dict)
#     print('cluster_point_dict', cluster_point_dict)
#     return json.dumps({
#         'gcluster_point_cluster_dict': point_cluster_dict,
#         'gcluster_cluster_point_dict': cluster_point_dict,
#     })

@app.route('/getClusterCenter', methods=['post'])
@cache.cached(timeout=0)
def get_cluster_center_coords():
    print('获取簇中心坐标：get_cluster_center_coords')
    data = request.get_json(silent=True)
    cluster_point_dict = data['cluster_point_dict']
    selected_cluster_idxs = data['selected_cluster_idxs']
    total_od_points = cache.get('total_od_points')

    tmp = {}
    for key in cluster_point_dict:
        tmp[int(key)] = cluster_point_dict[key]
    cluster_point_dict = tmp

    cid_center_coord_dict = get_cluster_center_coord(total_od_points, cluster_point_dict, selected_cluster_idxs)
    return json.dumps({
        'cid_center_coord_dict': cid_center_coord_dict
    })


@app.route('/getTrjDetail', methods=['post'])
def get_trj_detail():
    data = request.get_json(silent=True)
    # print(data)
    node_name = str(data['nodeName'])
    month = int(data['month'])
    start_day, end_day, start_hour, end_hour = [int(data['startDay']),
                                                int(data['endDay']),
                                                int(data['startHour']),
                                                int(data['endHour'])]
    cluster_point_dict = data['cluster_point_dict']
    force_nodes = data['force_nodes']
    tmp = {}
    for key in cluster_point_dict:
        tmp[int(key)] = cluster_point_dict[key]
    cluster_point_dict = tmp
    total_od_points = od_pair_process.get_od_points_filter_by_day_and_hour(month, start_day, end_day, 0, 24)['od_points']
    trj_idxs, node_names = get_trj_ids_by_force_node(force_nodes, cluster_point_dict, total_od_points)

    # ------------考虑一个 OD 对包含多个轨迹的情况，记录 nodeName 到 轨迹 ids 的映射-------------------
    node_name2trj_id = {}
    for i in range(len(trj_idxs)):
        if node_names[i] not in node_name2trj_id:
            node_name2trj_id[node_names[i]] = [trj_idxs[i]]
        else:
            node_name2trj_id[node_names[i]].append(trj_idxs[i])
    # -------------------------------------- -------------------------------------------------
    for key in node_name2trj_id.keys():
        print(key, node_name2trj_id[key])

    trj_ids = node_name2trj_id[node_name]
    ids, trips = get_trip_detail_by_id(trj_ids, month, start_day, end_day, start_hour, end_hour)
    od_gps_lst = []
    for trip in trips:
        od_gps_lst.append(trip[2])
        od_gps_lst.append(trip[-1])
    poi_id_file_id_dict = cache.get('poi_id_file_id_dict')
    kdtree = cache.get('kdtree')
    print('kdtree', kdtree)
    poi_info_lst = get_poi_info_lst_by_points(od_gps_lst, poi_id_file_id_dict, config_dict, kdtree, 300)
    print('poi_info_lst', poi_info_lst)
    print('trip num', len(trips))

    trj_detail = []
    trj_speed = []
    for i in range(len(trips)):
        # trj_detail.append([None for j in range(6)])
        one_trj_detail = {}
        one_trj_speed = []
        one_trj_detail['TrjId'] = trips[i][0][0]
        one_trj_detail['startPoint'] = f'{poi_info_lst[i*2]}'  # start point
        one_trj_detail['endPoint'] = f'{poi_info_lst[i*2+1]}'  # end point
        one_trj_detail['startTime'] = utils.timestamp_2time_format(trips[i][2][2])  # start time
        one_trj_detail['endTime'] = utils.timestamp_2time_format(trips[i][-1][2])  # end time
        mean_speed = 0
        for j in range(2, len(trips[i]) - 1):
            # 求平均速度，轨迹一共 (len(trips[i]) - 2 - 1) 段，-2 是减去 trjId 和 day，后面才是轨迹点，然后-1是段数
            speed = calTwoPointSpeed(trips[i][j], trips[i][j + 1])
            mean_speed += speed / (len(trips[i]) - 2 - 1)
            one_trj_speed.append(judgeSpeedLevel(speed))
        one_trj_detail['avgSpeed'] = f'{round(mean_speed, 2)} km/h'
        trj_detail.append(one_trj_detail)
        # trj_speed[one_trj_detail['TrjId']] = one_trj_speed
        trj_speed.append({
            'TrjId': one_trj_detail['TrjId'],
            'speedList':  one_trj_speed,
        })

    return json.dumps({
        'trj_detail': trj_detail,
        'trj_speed': trj_speed,
    })


@app.route('/getLineGraph', methods=['post'])
def get_line_graph():
    region = cache.get('region')
    data = request.get_json(silent=True)
    # print(data)
    month = int(data['month'])
    start_day, end_day, start_hour, end_hour = [int(data['startDay']),
                                                int(data['endDay']),
                                                int(data['startHour']),
                                                int(data['endHour'])]
    # selected_cluster_ids_in_brush = data['selectedClusterIdxsInBrush']
    # selected_cluster_ids = data['selectedClusterIdxs']
    out_adj_table = data['outAdjTable']
    cluster_point_dict = data['cluster_point_dict']
    with_space_dist = data['withSpaceDist']
    # with_space_dist: 是否考虑空间距离，如果不考虑，则把离散的点都连上 fake 边，保证它们聚集在一块，不会扩散到整个屏幕
    #  得到的 json 中 key 是 string，这里转成 int
    tmp = {}
    for key in out_adj_table:
        tmp[int(key)] = out_adj_table[key]
    out_adj_table = tmp
    # print(out_adj_table)
    # 计算线图，返回适用于 d3 的结构和邻接表 ===========================
    cid_center_coord_dict = get_cell_id_center_coord_dict(region)
    selected_cluster_ids = cid_center_coord_dict.keys()
    force_nodes, force_edges, filtered_adj_dict, lg = get_line_graph_by_selected_cluster(selected_cluster_ids, selected_cluster_ids, out_adj_table)

    #  计算簇中心坐标 ========================================
    tmp = {}
    for key in cluster_point_dict:
        tmp[int(key)] = cluster_point_dict[key]
    cluster_point_dict = tmp

    # total_od_points = cache.get('total_od_points')
    total_od_points = od_pair_process.get_od_points_filter_by_day_and_hour(month, start_day, end_day, 0, 24)['od_points']
    # cid_center_coord_dict = get_cluster_center_coord(total_od_points, cluster_point_dict, selected_cluster_ids)
    cache.set('cid_center_coord_dict', cid_center_coord_dict)

    #  计算 OD 对之间的距离 ==================================
    edge_name_dist_map = get_odpair_space_similarity(selected_cluster_ids, cid_center_coord_dict, force_nodes)

    #  将线图加入 fake 边，并给边添加距离，成为考虑 OD 对空间关系的线图 ===============================
    # if with_space_dist:
    force_edges = fuse_fake_edge_into_linegraph(force_nodes, force_edges, edge_name_dist_map)

    print('完全线图-点数', len(force_nodes))
    print('完全线图-边数', len(force_edges))

    # if not with_space_dist:
    #     force_edges = aggregate_single_points(force_nodes, force_edges, filtered_adj_dict)
    #     print('单独点聚合后-边数', len(force_edges))

    # +++++++++++++++ 轨迹获取和特征 ++++++++++++++
    # node_label_dict = None
    trj_idxs, node_names = get_trj_ids_by_force_node(force_nodes, cluster_point_dict, total_od_points)
    # ----------- 简单聚合，每个OD对取一个轨迹的特征---------
    tmp_trj_idx = []
    tmp_node_names = []
    name_set = {}
    for i in range(len(node_names)):
        if node_names[i] not in name_set:
            tmp_trj_idx.append(trj_idxs[i])
            tmp_node_names.append(node_names[i])
            name_set[node_names[i]] = 1
    # -------------------------------------------------
    trj_idxs, node_names = tmp_trj_idx, tmp_node_names
    print('按轨迹顺序排列的节点名的映射', node_names)
    # tid_trip_dict = get_trips_by_ids(trj_idxs, month, start_day, end_day)
    gps_trips = get_trips_by_ids(trj_idxs, month, start_day, end_day)
    # gps_trips = list(tid_trip_dict.values())
    feature = run_model2(args, gps_trips)
    # labels = get_cluster_by_trj_feature(args, feature)
    # print('labels', labels)
    # print('labels 数量', len(labels))
    # print('labels 全部类别数量', len(list(set(labels))))
    # print('labels 全部标签类别', list(set(labels)))
    node_label_dict = {}
    node_feature_dict = {}
    trjid_feature_dict = {}
    # for i in range(len(labels)):
    #     node_label_dict[node_names[i]] = labels[i]
    # 节点名称（${cid1}_${cid2}）和对应OD对特征的映射关系
    for i in range(len(feature)):
        node_feature_dict[node_names[i]] = feature[i]
        trjid_feature_dict[node_names[i]] = trj_idxs[i]
    print(f'轨迹id数{len(trj_idxs)}  轨迹数： {len(gps_trips)}  特征数: {len(feature)}  字典大小：{len(node_feature_dict.keys())} {len(node_names)}')
    # +++++++++++++++ 轨迹获取和特征 ++++++++++++++

    # ============== GCC 社区发现代码 ===============
    adj_mat = get_adj_matrix(lg)  # 根据线图得到 csc稀疏矩阵类型的邻接矩阵
    features, related_node_names = get_feature_list(lg, node_feature_dict)  # 根据线图节点顺序，整理一个节点向量数组，以及对应顺序的node name
    tmp_trj_idxs = []
    #  得到features对应顺序的 trjIds
    for node_name in related_node_names:
        tmp_trj_idxs.append(trjid_feature_dict[node_name])
    print(f'线图节点个数：{len(lg.nodes())}, 向量个数：{len(features)}')
    print('向量长度', len(features[0]))
    trj_labels = run(adj_mat, features)  # 得到社区划分结果，索引对应 features 的索引顺序，值是社区 id
    trj_labels = trj_labels.numpy().tolist()
    print(list(trj_labels))
    cluster_point_dict = {}
    for i in range(len(trj_labels)):
        label = trj_labels[i]
        if label not in cluster_point_dict:
            cluster_point_dict[label] = []
        # 在线图中度为 0 的散点，视为噪声，从社区中排除
        if get_degree_by_node_name(lg, node_names[i]) > 0:
            cluster_point_dict[label].append(node_names[i])
    print('实际社区个数: ', len(cluster_point_dict.keys()))
    # dag_force_nodes, dag_force_edges = get_dag_from_community(cluster_point_dict, force_nodes)

    # draw_cluster_in_trj_view(trj_labels, gps_trips)
    tsne_points = utils.DoTSNE(features, 2, cluster_point_dict)
    print('tsne_points', tsne_points)

    # ============== GCC 社区发现代码 ===============

    # ============== SA-Cluster 社区发现代码 ===============
    # # 为 line graph 添加属性，目前属性是随意值 TODO：属性改成轨迹特征聚类后的簇id，聚合成的一个整数　value
    # lg = update_graph_with_attr(lg, node_label_dict)
    # # 对线图进行图聚类，得到社区发现
    # point_cluster_dict, cluster_point_dict = get_cluster(lg, 7)
    # print('社区发现结果：')
    # print('point_cluster_dict', point_cluster_dict)
    # print('cluster_point_dict', cluster_point_dict)
    # #  将带属性的线图 networkx 对象存在全局缓存中
    # cache.set('line_graph', lg, timeout=0)
    # ============== 社区发现代码 ===============

    return json.dumps({
        'force_nodes': force_nodes,
        'force_edges': force_edges,
        'filtered_adj_dict': filtered_adj_dict,
        'cid_center_coord_dict': cid_center_coord_dict,
        'community_group': cluster_point_dict,
        'tsne_points': tsne_points,
        'trj_labels': trj_labels,  # 每个节点（OD对）的社区label，与 tsne_points 顺序对应
        'related_node_names': related_node_names,
        'tmp_trj_idxs': tmp_trj_idxs,  # 与 tsne_points 顺序对应
        # 'tid_trip_dict': tid_trip_dict,
    })


@app.route('/getPoiInfoByPoint', methods=['post'])
@cache.cached(timeout=0)
def get_poi_info_by_point():
    data = request.get_json(silent=True)
    point_in_cluster, radius = data['point_in_cluster'], int(data['radius'])

    poi_id_file_id_dict = cache.get('poi_id_file_id_dict')
    kdtree = cache.get('kdtree')
    poi_type_dict = get_poi_type_filter_by_radius(point_in_cluster, poi_id_file_id_dict, config_dict, kdtree, radius)
    return json.dumps({
        'poi_type_dict': poi_type_dict,
    })


@app.route('/getGccDataVis', methods=['get'])
def get_gcc_data_vis():
    dataset = ['cora', 'citeseer', 'pubmed', 'wiki']
    data = sio.loadmat(os.path.join('/home/zhengxuan.lin/project/od_trajectory_analize/backend/datasetVis/', f'{dataset[2]}.mat'))
    print(data.keys())
    adj = data['W']
    adj = adj.astype(float)
    if not sp.issparse(adj):
        adj = sp.csc_matrix(adj)

    force_nodes = []
    force_edges = []
    st = set()
    row, col = adj.nonzero()
    id_lst = []
    for i in range(len(row)):
        if random.random() < 0.05:
            id_lst.append(i)
            force_edges.append({'source': int(row[i]), 'target': int(col[i])})
            st.add(int(row[i]))
            st.add(int(col[i]))
    for i in range(len(row)):
        if i in st:
            force_nodes.append({'name': i})

    print(f'node num: {len(force_nodes)}', f'edge num: {len(force_edges)}')

    G = []
    if os.path.exists('/home/zhengxuan.lin/project/od_trajectory_analize/backend/gcc/graph_convolutional_clustering/data/gcc_G.pkl'):
        print('exists trained G')
        with open("/home/zhengxuan.lin/project/od_trajectory_analize/backend/gcc/graph_convolutional_clustering/data/gcc_G.pkl", 'rb') as f:
            obj = pickle.loads(f.read())
            G = obj['G']
            G = G.numpy().tolist()

    # force_nodes = [{'name': 'A'}, {'name': 'B'}, {'name': 'C'}]
    # force_edges = [{'source': 'A', 'target': 'C'}, {'source': 'A', 'target': 'B'}]

    return json.dumps({
        'force_nodes': force_nodes,
        'force_edges': force_edges,
        'id_lst': id_lst,
        'G': G,
    })


@app.before_first_request
def runserver():
    print('-----------====================================================')
    start_time = datetime.now()
    with open("/home/zhengxuan.lin/project/od_trajectory_analize/backend/data/POI映射关系.pkl", 'rb') as f:
        obj = pickle.loads(f.read())
        total_poi_coor = obj['total_poi_coor']
        file_id_poi_id_dict = obj['file_id_poi_id_dict']
        poi_id_file_id_dict = obj['poi_id_file_id_dict']
        kdtree = obj['kdtree']
        print('kdtree', kdtree)
        cache.set('file_id_poi_id_dict', file_id_poi_id_dict)
        cache.set('poi_id_file_id_dict', poi_id_file_id_dict)
        cache.set('total_poi_coor', total_poi_coor)
        cache.set('kdtree', kdtree)
        print('total_poi_coor', kdtree)
        print('读取POI文件结束，用时: ', (datetime.now() - start_time))
#     # do something before running the server
#     _thread.start_new_thread(app.run(port=5050, host='0.0.0.0'))


if __name__ == '__main__':
    # runserver()
    print('========>', os.getcwd())
    # test()
    region = get_region()
    cache.set('region', region)
    #==_asdfas2
    _thread.start_new_thread(app.run(port=5000, host='0.0.0.0'))
    # start_time = datetime.now()
    # print('-----------====================================================')
    # with open("/home/zhengxuan.lin/project/od_trajectory_analize/backend/data/POI映射关系.pkl", 'rb') as f:
    #     obj = pickle.loads(f.read())
    #     total_poi_coor = obj['total_poi_coor']
    #     file_id_poi_id_dict = obj['file_id_poi_id_dict']
    #     poi_id_file_id_dict = obj['poi_id_file_id_dict']
    #     kdtree = obj['kdtree']
    #     print('kdtree', kdtree)
    #     cache.set('file_id_poi_id_dict', file_id_poi_id_dict)
    #     cache.set('poi_id_file_id_dict', poi_id_file_id_dict)
    #     cache.set('total_poi_coor', total_poi_coor)
    #     cache.set('kdtree', kdtree)
    #     print('total_poi_coor', kdtree)
    #     print('读取POI文件结束，用时: ', (datetime.now() - start_time))

