from flask import Flask
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
    return od_pair_process.get_total_od_points()


if __name__ == '__main__':
    app.run()
