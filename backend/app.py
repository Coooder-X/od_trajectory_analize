import json

from flask import Flask, request
from flask_cors import CORS

from data_process import od_pair_process

app = Flask(__name__)
CORS(app, resources=r'/*')
#  python -m flask run


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
    return json.dumps(od_pair_process.get_od_points_filter_by_hour(start_hour, end_hour))


@app.route('/clutering', methods=['get', 'post'])
def get_cluster_result(k, theta):
    return []


if __name__ == '__main__':
    app.run()
