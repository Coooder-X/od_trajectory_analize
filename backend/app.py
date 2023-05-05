import json
from datetime import datetime

import numpy as np
from flask import Flask, request
from flask_caching import Cache
from flask_cors import CORS
import _thread

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
    start_day, end_day, start_hour, end_hour = request.args.get('startDay', type=int), \
                                               request.args.get('endDay', type=int), \
                                               request.args.get('startHour', 0, type=int), \
                                               request.args.get('endHour', 24, type=int)
    print(start_day, end_day, start_hour, end_hour)
    # return json.dumps({'od_points':od_pair_process.get_hour_od_points()})
    return json.dumps(od_pair_process.get_od_points_filter_by_day_and_hour(start_day, end_day, start_hour, end_hour))


@app.route('/getTrjNum', methods=['get'])
def get_trj_num():
    start_day, end_day, start_hour, end_hour = request.args.get('startDay', type=int), \
                                               request.args.get('endDay', type=int), \
                                               request.args.get('startHour', 0, type=int), \
                                               request.args.get('endHour', 24, type=int)
    print(start_day, end_day, start_hour, end_hour)
    # return json.dumps({'od_points':od_pair_process.get_hour_od_points()})
    x = od_pair_process.get_trj_num_filter_by_day_and_hour(start_day, end_day, start_hour, end_hour)
    trj_num = len(od_pair_process.get_trj_num_filter_by_day_and_hour(start_day, end_day, start_hour, end_hour)['trips'])
    print(trj_num)
    return json.dumps({'trj_num': trj_num})


@app.route('/getTrjNumByHour', methods=['get'])
def get_trj_num_by_hour():
    date = request.args.get('date', type=int)
    print("输入参数为%d日" % date)
    res = od_pair_process.trj_num_by_hour(date)
    return json.dumps({'nums': res})


@app.route('/getTrjNumByOd', methods=['get'])
def get_trj_num_by_od():
    date, num = request.args.get('date', type=int), request.args.get('num', type=int)
    src_id_list, tgt_id_list = request.args.getlist('src_id_list'), request.args.getlist('tgt_id_list')
    total_od_points = get_od_points_filter_by_day_and_hour(date - num, date, 0, 24)
    res = []
    for d in range(date - num, date + 1):
        num = []
        for h in range(24):
            count = 0
            for src_id in src_id_list:
                for tgt_id in tgt_id:
                    src = total_od_points[src_id]
                    tgt = total_od_points[tgt]
                    if src[4] == 0 and tgt[4] == 1 and src[3] == tgt[3] and src[5] == d and tgt[5] == d and h * 24 <= \
                            src[2] <= (h + 1) * 24 and h * 24 <= tgt[2] <= (h + 1) * 24:
                        count = count + 1
            num.append(count)
        res.append(num)
    return res


@app.route('/getClusteringResult', methods=['get', 'post'])
def get_cluster_result():
    k, theta = int(request.args['k']), int(request.args['theta'])
    print(f'k={k}, theta:{theta}')
    start_hour, end_hour = int(request.args['startHour']), int(request.args['endHour'])

    start_time = datetime.now()
    od_points = np.asarray(od_pair_process.get_total_od_points())    # get_total_od_points
    total_od_coord_points = od_points[:, 0:2]  # 并去掉时间戳留下经纬度坐标
    print('读取OD点结束，用时: ', (datetime.now() - start_time))
    print('pos nums', len(od_points), '\n开始聚类')
    start_time = datetime.now()
    point_cluster_dict, cluster_point_dict = delaunay_clustering(k=k, theta=theta, od_points=total_od_coord_points)
    print('结束聚类，用时: ', (datetime.now() - start_time))

    res = od_pair_process.get_od_points_filter_by_hour(start_hour, end_hour)
    index_lst = res['index_lst']
    part_od_points = res['part_od_points']
    new_point_cluster_dict, new_cluster_point_dict = cluster_filter_by_hour(index_lst, point_cluster_dict)
    print('过滤后的点数：', len(index_lst))
    print('过滤后的的簇数：', len(new_cluster_point_dict.keys()))
    out_adj_table, in_adj_table = build_od_graph(new_point_cluster_dict, od_points, index_lst)
    print(out_adj_table)
    # draw_DT_clusters(new_cluster_point_dict, total_od_coord_points, k, theta, start_hour, end_hour, set(index_lst))
    return json.dumps({
        'index_lst': index_lst,
        'point_cluster_dict': point_cluster_dict,
        'cluster_point_dict': new_cluster_point_dict,
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

    cid_center_coord_dict = get_cluster_center_coord(cluster_point_dict, selected_cluster_ids)
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


if __name__ == '__main__':
    _thread.start_new_thread(app.run(host='0.0.0.0'))
