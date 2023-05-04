import json
from datetime import datetime

import numpy as np
from flask import Flask, request
from flask_cors import CORS
import _thread

from data_process.OD_area_graph import build_od_graph, get_line_graph_by_selected_cluster
from data_process import od_pair_process
from data_process.DT_graph_clustering import delaunay_clustering, cluster_filter_by_hour, draw_DT_clusters

app = Flask(__name__)
CORS(app, resources=r'/*')


#  python -m flask run

# out_adj_dict = {}


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/getTotalODPoints', methods=['get', 'post'])
def get_total_od_points():
    return json.dumps(od_pair_process.get_hour_od_points())


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
    x =  od_pair_process.get_od_points_filter_by_day_and_hour(start_day, end_day, start_hour, end_hour)
    return json.dumps(od_pair_process.get_od_points_filter_by_day_and_hour(start_day, end_day, start_hour, end_hour))


@app.route('/getTrjNum', methods=['get'])
def get_trj_num():
    start_day, end_day, start_hour, end_hour = request.args.get('startDay', type=int), \
                                               request.args.get('endDay', type=int), \
                                               request.args.get('startHour', 0, type=int), \
                                               request.args.get('endHour', 24, type=int)
    print(start_day, end_day, start_hour, end_hour)
    # return json.dumps({'od_points':od_pair_process.get_hour_od_points()})
    trj_num = len(od_pair_process.get_od_points_filter_by_day_and_hour(start_day, end_day, start_hour, end_hour)['od_points'])
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
    part_od_points = res['od_points']
    new_point_cluster_dict, new_cluster_point_dict = cluster_filter_by_hour(index_lst, point_cluster_dict)
    print('过滤后的点数：', len(index_lst))
    print('过滤后的的簇数：', len(new_cluster_point_dict.keys()))
    json_adj_table, json_nodes, out_adj_table, in_adj_table = build_od_graph(new_point_cluster_dict, od_points, index_lst)
    # draw_DT_clusters(new_cluster_point_dict, total_od_coord_points, k, theta, start_hour, end_hour, set(index_lst))
    return json.dumps({
        'index_lst': index_lst,
        'point_cluster_dict': point_cluster_dict,
        'cluster_point_dict': new_cluster_point_dict,
        'od_points': part_od_points,
        'json_adj_table': json_adj_table,
        'json_nodes': json_nodes,
        'out_adj_table': out_adj_table,
        'in_adj_table': in_adj_table,
    })


@app.route('/getLineGraph', methods=['post'])
def get_line_graph():
    data = request.get_json(silent=True)
    # print(data)
    selected_cluster_ids = data['selectedClusterIdxs']
    out_adj_table = data['outAdjTable']
    #  得到的 json 中 key 是 string，这里转成 int
    tmp = {}
    for key in out_adj_table:
        tmp[int(key)] = out_adj_table[key]
    out_adj_table = tmp
    force_nodes, force_edges, filtered_adj_dict = get_line_graph_by_selected_cluster(selected_cluster_ids, out_adj_table)
    return json.dumps({
        'force_nodes': force_nodes,
        'force_edges': force_edges,
        'filtered_adj_dict': filtered_adj_dict,
    })


if __name__ == '__main__':
    _thread.start_new_thread(app.run(host='0.0.0.0'))
