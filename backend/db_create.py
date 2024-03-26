from global_param import db_name
from database.table_create import get_db_connection, init_database, create_trj_table, delete_trj_table, get_trips_by_day, create_od_table


def init_trj_table():
    delete_trj_table(db_name)
    get_db_connection()
    init_database(db_name)
    create_trj_table(db_name)


if __name__ == '__main__':
    # ========= 初始化轨迹表 ===========
    # init_trj_table()
    # get_trips_by_day('trajectory_db', 1, 2)  # test
    # ========= 初始化 OD 表 ===========
    # create_od_table('trajectory_db')
