import json
import pickle
from datetime import datetime

import numpy as np
from flask import Flask, request
from flask_caching import Cache
from flask_cors import CORS
import _thread
import utils
import os

from poi_process.new_read_poi import get_poi_type_filter_by_radius, config_dict, getPOI_Coor, buildKDTree, meters2lonlat_list, lonlat2meters_poi
from data_process.od_pair_process import get_odpair_space_similarity
from graph_cluster_test.sa_cluster import update_graph_with_attr, get_cluster
from data_process.OD_area_graph import build_od_graph, get_line_graph_by_selected_cluster, get_cluster_center_coord, \
    fuse_fake_edge_into_linegraph  #, aggregate_single_points
from data_process import od_pair_process
from data_process.DT_graph_clustering import delaunay_clustering, cluster_filter_by_hour, draw_DT_clusters

app = Flask(__name__)
CORS(app, resources=r'/*')


#  python -m flask run

# out_adj_dict = {}
line_graph = None
app.config["CACHE_TYPE"] = "simple"
cache = Cache(app)


@app.route('/')
def hello_world():  # put application's code here
    start_time = datetime.now()
    with open("/home/linzhengxuan/project/od_trajectory_analize/backend/data/POI映射关系.pkl", 'rb') as f:
        obj = pickle.loads(f.read())
        total_poi_coor = obj['total_poi_coor']
        file_id_poi_id_dict = obj['file_id_poi_id_dict']
        poi_id_file_id_dict = obj['poi_id_file_id_dict']
        kdtree = obj['kdtree']
        cache.set('file_id_poi_id_dict', file_id_poi_id_dict)
        cache.set('poi_id_file_id_dict', poi_id_file_id_dict)
        cache.set('total_poi_coor', total_poi_coor)
        cache.set('kdtree', kdtree)
        print('total_poi_coor', kdtree)
        print('读取POI文件结束，用时: ', (datetime.now() - start_time))
    # start_time = datetime.now()
    # total_poi_coor, file_id_poi_id_dict, poi_id_file_id_dict = getPOI_Coor(config_dict['poi_dir'])
    # total_poi_coor = lonlat2meters_poi(total_poi_coor)
    # kdtree = buildKDTree(total_poi_coor)
    # with open("/home/linzhengxuan/project/od_trajectory_analize/backend/data/POI映射关系.pkl", 'wb') as f:
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
    data_path = "/tmp/" + str(month).zfill(2) + "trj_num_by_month.txt"
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


@app.route('/getTrjNumByOd', methods=['get'])
def get_trj_num_by_od():
    month = request.args.get('month', 5, type=int)
    date, num = request.args.get('date', type=int), request.args.get('num', type=int)
    src_id_list, tgt_id_list = request.args.getlist('src_id_list'), request.args.getlist('tgt_id_list')
    total_od_points = od_pair_process.get_od_points_filter_by_day_and_hour(month, date - num, date, 0, 24)['od_points']
    res = []
    for d in range(date - num, date + 1):
        num = []
        for h in range(24):
            count = 0
            for src_id in src_id_list:
                for tgt_id in tgt_id_list:
                    src = total_od_points[src_id]
                    tgt = total_od_points[tgt_id]
                    if src[4] == 0 and tgt[4] == 1 and src[3] == tgt[3] and src[5] == d and tgt[5] == d and h * 24 <= \
                            src[2] <= (h + 1) * 24:
                        count = count + 1
            num.append(count)
        res.append(num)
    return res


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

@app.route('/getClusteringResult', methods=['get', 'post'])
def get_cluster_result():
    month = int(request.args['month'])
    k, theta = int(request.args['k']), int(request.args['theta'])
    print(f'k={k}, theta:{theta}')
    start_day, end_day = int(request.args['startDay']), int(request.args['endDay'])
    start_hour, end_hour = int(request.args['startHour']), int(request.args['endHour'])

    start_time = datetime.now()
    # od_points = np.asarray(od_pair_process.get_total_od_points())    # get_total_od_points
    res = od_pair_process.get_od_points_filter_by_day_and_hour(month, start_day, end_day)
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
    print(out_adj_table)
    # draw_DT_clusters(new_cluster_point_dict, total_od_coord_points, k, theta, start_hour, end_hour, set(index_lst))

    total_cluster_point_dict = {}
    for key in cluster_point_dict:
        total_cluster_point_dict[key] = list(cluster_point_dict[key])
    return json.dumps({
        'index_lst': index_lst,
        'point_cluster_dict': point_cluster_dict,
        'cluster_point_dict': total_cluster_point_dict,
        'part_cluster_point_dict': new_cluster_point_dict,
        'part_od_points': part_od_points,
        # 'json_adj_table': json_adj_table,
        # 'json_nodes': json_nodes,
        'out_adj_table': out_adj_table,
        'in_adj_table': in_adj_table,
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
def get_cluster_center_coords():
    print('获取簇中心坐标：get_cluster_center_coords')
    data = request.get_json(silent=True)
    cluster_point_dict = data['cluster_point_dict']
    selected_cluster_idxs = data['selected_cluster_idxs']

    tmp = {}
    for key in cluster_point_dict:
        tmp[int(key)] = cluster_point_dict[key]
    cluster_point_dict = tmp

    cid_center_coord_dict = get_cluster_center_coord(cluster_point_dict, selected_cluster_idxs)
    return json.dumps({
        'cid_center_coord_dict': cid_center_coord_dict
    })


@app.route('/getLineGraph', methods=['post'])
def get_line_graph():
    data = request.get_json(silent=True)
    # print(data)
    selected_cluster_ids_in_brush = data['selectedClusterIdxsInBrush']
    selected_cluster_ids = data['selectedClusterIdxs']
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
    force_nodes, force_edges, filtered_adj_dict, lg = get_line_graph_by_selected_cluster(selected_cluster_ids_in_brush, selected_cluster_ids, out_adj_table)

    #  计算簇中心坐标 ========================================
    tmp = {}
    for key in cluster_point_dict:
        tmp[int(key)] = cluster_point_dict[key]
    cluster_point_dict = tmp

    total_od_points = cache.get('total_od_points')
    cid_center_coord_dict = get_cluster_center_coord(total_od_points, cluster_point_dict, selected_cluster_ids)
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

    # ============== 社区发现代码 ===============
    # 为 line graph 添加属性，目前属性是随意值 TODO：属性改成轨迹特征聚类后的簇id，聚合成的一个整数　value
    lg = update_graph_with_attr(lg)
    # 对线图进行图聚类，得到社区发现
    point_cluster_dict, cluster_point_dict = get_cluster(lg, 8)
    print('point_cluster_dict', point_cluster_dict)
    print('cluster_point_dict', cluster_point_dict)
    #  将带属性的线图 networkx 对象存在全局缓存中
    cache.set('line_graph', lg, timeout=0)
    # ============== 社区发现代码 ===============

    return json.dumps({
        'force_nodes': force_nodes,
        'force_edges': force_edges,
        'filtered_adj_dict': filtered_adj_dict,
        'cid_center_coord_dict': cid_center_coord_dict,
        'community_group': cluster_point_dict,
    })


@app.route('/getPoiInfoByPoint', methods=['post'])
def get_poi_info_by_point():
    data = request.get_json(silent=True)
    point_in_cluster, radius = data['point_in_cluster'], int(data['radius'])

    poi_id_file_id_dict = cache.get('poi_id_file_id_dict')
    kdtree = cache.get('kdtree')
    poi_type_dict = get_poi_type_filter_by_radius(point_in_cluster, poi_id_file_id_dict, config_dict, kdtree, radius)
    return json.dumps({
        'poi_type_dict': poi_type_dict,
    })


if __name__ == '__main__':
    _thread.start_new_thread(app.run(host='0.0.0.0'))
