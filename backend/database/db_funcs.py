import json
from datetime import datetime

from database.table_create import get_trips_by_day
from database.test import get_db_connection

from data_process.SpatialRegionTools import inregionS

from global_param import db_name


def query_trips_by_day(db_name, start_day, end_day):
    return get_trips_by_day(db_name, start_day, end_day)


def query_trj_by_day_and_hour(month, start_day, end_day, start_hour, end_hour):
    """
    给定时间范围，查询出发时间在该范围内轨迹。
    """
    table_name = 'trj_table'
    db = get_db_connection(db_name)
    cursor = db.cursor()
    sql = f"SELECT * FROM {table_name} " \
          f"where date >= {start_day} and date < {end_day} " \
          f"and month={month} " \
          f"and start_time >= {start_hour * 3600} and start_time <= {end_hour * 3600};"
    print(f'[INFO] 执行 sql, 查询 {month} 月 [{start_day}, {end_day}) 日间，[{start_hour}, {end_hour}] 时内的轨迹: {sql}')
    cursor.execute(sql)

    trips = cursor.fetchall()  # fetchall() 获取所有记录
    res = []
    for (i, trip) in enumerate(trips):
        day, point_list = trip[2], json.loads(trip[5])
        cur_trip = [[i], [day]]
        cur_trip.extend(point_list)
        # 轨迹数据结构：[[index], [date], [lon1, lat1, time1], ..., [lon, lat, time]]
        res.append(cur_trip)
    index_list = range(len(trips))
    return res, index_list


def query_od_by_trj_day_and_hour(month, start_day, end_day, start_hour, end_hour, region):
    """
    给定时间范围，查询出发、到达时间均在该范围内轨迹。根据查询到的轨迹，获取轨迹头尾的采样点，作为 OD 对。
    用于构建线图逻辑中，计算 total_od_pairs 的部分。
    """
    table_name = 'trj_table'
    db = get_db_connection(db_name)
    cursor = db.cursor()
    # 获取的是起始、结束时间都在给定时间范围内的轨迹！
    sql = f"SELECT point_list FROM {table_name} " \
          f"where date >= {start_day} and date < {end_day} " \
          f"and month={month} " \
          f"and start_time >= {start_hour * 3600} and end_time <= {end_hour * 3600};"
    print(f'[INFO] 执行 sql, 查询 {month} 月 [{start_day}, {end_day}) 日间，严格处于 [{start_hour}, {end_hour}] 时内的轨迹: {sql}')
    cursor.execute(sql)

    trips = cursor.fetchall()  # fetchall() 获取所有记录
    res = []
    for trip in trips:
        point_list = json.loads(trip[0])
        o, d = point_list[0], point_list[-1]
        if inregionS(region, o[0], o[1]) and inregionS(region, d[0], d[1]):
            res.append([o, d])
    return res


def query_od_points_by_day_and_hour(month, start_day, end_day, start_hour, end_hour):
    """
    给定时间范围，查询出发时间在该范围内轨迹的 OD 点，返回的数据用于前端数据视图确定参数后，展示GIS视图中的OD点
    """
    table_name = 'trj_table'
    db = get_db_connection(db_name)
    cursor = db.cursor()
    # 只要 O 点的时间在范围内即可，D 点也加入进去
    sql = f"SELECT date, point_list FROM {table_name} " \
          f"where date >= {start_day} and date <= {end_day} " \
          f"and month={month} " \
          f"and start_time >= {start_hour * 3600} and start_time <= {end_hour * 3600};"
    print(f'[INFO] 执行 sql, 查询 {month} 月 [{start_day}, {end_day}] 日间，出发时间在 [{start_hour}, {end_hour}] 时内的轨迹: {sql}')
    t = datetime.now()
    cursor.execute(sql)

    trips = cursor.fetchall()  # fetchall() 获取所有记录
    print(f'[INFO] =====> 查询轨迹数据耗时 {datetime.now() - t}')
    res = []
    index_list = []
    t = datetime.now()
    for index, trip in enumerate(trips):
        point_list = json.loads(trip[1])
        o, d = point_list[0], point_list[-1]
        day = trip[0]
        lon, lat, time = o[0], o[1], o[2]
        # 此处的 OD 点数据结构沿用了旧的结构，与数据库中的不相同
        res.append([lon, lat, time, index, 0, day])
        index_list.append(index * 2)
        lon, lat, time = d[0], d[1], d[2]
        res.append([lon, lat, time, index, 1, day])
        index_list.append(index * 2 + 1)

    print(f'[INFO] =====> 处理轨迹数据耗时 {datetime.now() - t}')
    print(f'[INFO] {start_day}-{end_day} {start_hour}-{end_hour} OD点总数：', len(res))
    return {'od_points': res, 'index_lst': index_list}


def query_trj_num_by_day(month, day):
    table_name = 'trj_table'
    db = get_db_connection(db_name)
    cursor = db.cursor()
    # 只要 O 点的时间在范围内即可，D 点也加入进去
    sql = f"SELECT COUNT(*) FROM {table_name} " \
          f"where date = {day} " \
          f"and month={month};"
    print(f'[INFO] 执行 sql, 查询{month}月{day}日的轨迹数量: {sql}')
    cursor.execute(sql)
    count = cursor.fetchall()[0][0]  # fetchall() 获取所有记录
    print(f'[INFO] res = {count}')
    return count
