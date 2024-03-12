import json
import pickle

import h5py
import mysql.connector
import numpy as np
from data_process.SpatialRegionTools import get_cell_id_center_coord_dict, makeVocab, inregionS


def get_db_connection(db_name=None):
    if db_name is None:
        db = mysql.connector.connect(
            host="10.105.16.22",  # 数据库主机地址
            port=3336,
            user="root",  # 数据库用户名
            passwd="root"  # 数据库密码
        )
    else:
        db = mysql.connector.connect(
            host="10.105.16.22",  # 数据库主机地址
            port=3336,
            user="root",  # 数据库用户名
            passwd="root",  # 数据库密码
            database=db_name
        )
    return db


def init_database(db_name):
    db = get_db_connection()

    print(f'数据库：{db}')
    mycursor = db.cursor()

    mycursor.execute("SHOW DATABASES")

    if (db_name,) not in mycursor:
        print(f'数据库 {db_name} 不存在，执行 CREATE DATABASE {db_name};')
        mycursor.execute(f"CREATE DATABASE {db_name}")
        mycursor.execute("SHOW DATABASES")
        for x in mycursor:
            print(x)
    else:
        print(f'数据库 {db_name} 已存在')


# 定义一个检查表的函数
def check_table_exists(db_name, table_name):
    db = get_db_connection(db_name)
    # 初始化游标对象
    cursor = db.cursor()
    # 执行 SHOW TABLES 查询
    cursor.execute("SHOW TABLES")
    # 获取查询结果
    tables = cursor.fetchall()
    # 遍历结果，判断表名是否存在
    for table in tables:
        if table[0] == table_name:
            return True
    return False


def delete_trj_table(db_name):
    db = get_db_connection(db_name)
    mycursor = db.cursor()
    mycursor.execute('DROP TABLE IF EXISTS trj_table;')
    print('删除该数据表...')


def create_trj_table(db_name):
    db = get_db_connection(db_name)
    mycursor = db.cursor()
    if check_table_exists(db_name, 'trj_table'):
        # 执行 SQL 语句，使用 COUNT 函数来统计表中的记录数
        mycursor.execute("SELECT COUNT(*) FROM trj_table;")
        # 获取查询结果，并打印或返回
        count = mycursor.fetchone()[0]
        print(f'数据库 {db_name} 中，表 trj_table 已存在，数据行数为 {count}')
        if count > 0:
            return
        else:
            print('删除重建该数据表...')
            mycursor.execute('DROP TABLE IF EXISTS trj_table;')
    print(f'数据库 {db_name} 中，表 trj_table 正在创建...')
    mycursor.execute('CREATE TABLE trj_table (' +
                     'id INT AUTO_INCREMENT PRIMARY KEY,' +
                     'month INT,' +
                     'date INT,' +
                     'start_time INT DEFAULT NULL,' +
                     'end_time INT DEFAULT NULL,' +
                     'point_list JSON DEFAULT NULL' +
                     ');'
                     )
    print('数据库创建成功，开始初始化轨迹数据...')

    with open("/home/zhengxuan.lin/project/od_trajectory_analize/backend/data/region.pkl", 'rb') as file:
        trj_region = pickle.loads(file.read())
        print('是否考虑轨迹时间：', trj_region.needTime)
        needTime = True

    start_day, end_day, month = 1, 31, 5
    for day in range(start_day, end_day + 1):
        print(f'正在插入第 {day} 天的轨迹数据...')
        # data_target_path = "/home/zhengxuan.lin/project/tmp/" + "2020" + str(month).zfill(2) + str(i).zfill(2) + "_trj.pkl"
        data_source_path = "/home/zhengxuan.lin/project/" + str(month) + "月/" + str(month).zfill(2) + "月" + str(
            day).zfill(
            2) + "日/2020" + str(month).zfill(2) + str(day).zfill(
            2) + "_hz.h5"
        with h5py.File(data_source_path, 'r') as f:
            print('轨迹数: ', len(f['trips']))
            trips = []
            lines = []
            for i in range(0, len(f['trips'])):
                locations = f['trips'][str(i + 1)]
                trip = []
                line = []
                if needTime:
                    timestamp = f["timestamps"][str(i + 1)]
                    for ((lon, lat), time) in zip(locations, timestamp):
                        time = int(time)
                        trip.append([lon, lat, time])
                else:
                    for (lon, lat) in locations:
                        trip.append([lon, lat])
                # print(trip)
                for j in range(len(trip) - 1):
                    line.append([(trip[j][0], trip[j][1]), (trip[j + 1][0], trip[j + 1][1])])

                date = day
                point_list = str(json.dumps(trip))
                # print('point_list', point_list)

                if needTime:  # false, why?
                    start_time, end_time = trip[0][2], trip[-1][2]
                    # sql = f"-- INSERT INTO trj_table (month, date, start_time, end_time, 'point_list') VALUES (%d, %d, %d, %d, %s)"
                    sql = f"INSERT INTO trj_table (month, date, start_time, end_time, point_list) VALUES ({month}, {date}, {start_time}, {end_time}, '{point_list}');"
                else:
                    sql = f"INSERT INTO trj_table (month, date, start_time, end_time, point_list) VALUES ({month}, {date}, NULL, NULL, '{point_list}');"
                mycursor.execute(sql)

                lines.append(line)
                trip = np.array(trip)
                trips.append(trip)
                if len(trips) % 10000 == 0:
                    print(f'写入{len(trips)}条轨迹...')
        db.commit()  # 数据表内容有更新，必须使用到该语句
    return trips, lines


def get_trips_by_day(db_name, start_day, end_day):
    db = get_db_connection(db_name)
    cursor = db.cursor()
    cursor.execute(f"SELECT * FROM trj_table where date >= {start_day} and date < {end_day}")

    trips = cursor.fetchall()  # fetchall() 获取所有记录
    res = []
    for trip in trips:
        trip = list(trip)
        trip[5] = json.loads(trip[5])
        res.append(tuple(trip))
    return res


def create_od_table(db_name):
    table_name = 'od_table'
    db = get_db_connection(db_name)
    cursor = db.cursor()
    if check_table_exists(db_name, table_name):
        print(f'数据库 {db_name} 中，表 {table_name} 已存在')
        print('删除重建该数据表...')
        cursor.execute(f'DROP TABLE IF EXISTS {table_name};')
        # return

    print(f'数据库 {db_name} 中，表 {table_name} 正在创建...')
    cursor.execute(f'CREATE TABLE {table_name} (' +
                     'id INT AUTO_INCREMENT PRIMARY KEY,' +
                     'trj_id INT,' +
                     'month INT,' +
                     'date INT,' +
                     'lon DOUBLE DEFAULT NULL,' +
                     'lat DOUBLE DEFAULT NULL,' +
                     'time INT DEFAULT NULL,' +
                     'flag BIT' +
                     ');'
                     )
    start_day, end_day, month = 1, 31, 5
    for day in range(start_day, end_day + 1):
        print(f'正在插入第 {day} 天的 OD 数据...')
        today_trips = get_trips_by_day(db_name, day, day + 1)
        for index, trip in enumerate(today_trips):
            if index % 10000 == 0:
                print(f'已写入{index}条轨迹的 OD 点...')
            trj_id, m, d, _, _, point_list = trip
            o_lon, o_lat, o_time = point_list[0][0], point_list[0][1], point_list[0][2]
            d_lon, d_lat, d_time = point_list[-1][0], point_list[-1][1], point_list[-1][2]
            sql = f"INSERT INTO {table_name} (trj_id, month, date, lon, lat, time, flag)" \
                  f" VALUES ({trj_id}, {month}, {day}, {o_lon}, {o_lat}, {o_time}, {0});"
            cursor.execute(sql)
            sql = f"INSERT INTO {table_name} (trj_id, month, date, lon, lat, time, flag)" \
                  f" VALUES ({trj_id}, {month}, {day}, {d_lon}, {d_lat}, {d_time}, {1});"
            cursor.execute(sql)
        print(f'第 {day} 天的 OD 数据插入完成')
        db.commit()  # 数据表内容有更新，必须使用到该语句


# if __name__ == '__main__':
#     get_db_connection()
#     db_name = 'trajectory_db'
#     init_database(db_name)
#     create_trj_table(db_name)
