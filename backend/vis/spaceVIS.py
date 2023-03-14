import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import random
import numpy as np
import math
import pickle
from data_process.SpatialRegionTools import gps2vocab, gps2cell, cell2coord

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def trip2seq(region, trj_data):
    seq = []
    for (lon, lat) in trj_data:
        # 不在范围的点会变成UNK
        seq.append(gps2vocab(region, lon, lat))
    return seq

colorsOfLabel = [
    "#8ECFC9","#2878b5","#F27970","#A1A9D0","#C4A5DE","#63b2ee","#9192ab","#3b6291","#bf7334","#3b8ba1","#ffbd66",
    "#FFBE7A","#9ac9db","#BB9727","#F0988C","#F6CAE5","#76da91","#7898e1","#943c39","#3f6899","#c97937","#f74d4d",
    "#FA7F6F","#f8ac8c","#54B345","#B883D4","#96CCCB","#f8cb7f","#efa666","#779043","#9c403d","#002c53","#2455a4",
    "#82B0D2","#c82423","#32B897","#9E9E9E","#8983BF","#f89588","#eddd86","#624c7c","#7d9847","#ffa510","#41b7ac",
    "#BEB8DC","#ff8884","#05B9E2","#CFEAF1","#C76DA2","#7cd6cf","#9987ce","#388498","#675083","#0c84c6","#E7DAD2",
    "#63b2ee","#76da91"
]

if __name__ == "__main__":
    with open("../data/region.pkl", 'rb') as file:
        region = pickle.loads(file.read())

    fig = plt.figure(figsize=(20, 10))
    ax = fig.subplots()
    # ax = fig.add_subplot(111)
    # ax = Axes3D(fig)

    # colorsOfLabel = {}
    # colorsOfNeighbor = {}
    # for i in range(52): # label num
    #     colorsOfLabel[i] = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])

    # for i in range(20): # k
    #     colorsOfNeighbor[i] = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
    # with h5py.File('./preprocessing/experiment_data/hzd2zjg_reorder_3_inte.h5', 'r') as f:
    with h5py.File('../make_data/20200101_jianggan.h5', 'r') as f:
        with open("../data/train.csv", 'r') as labelF:
            with h5py.File("../data/hangzhou-vocab-dist-cell40.h5") as kf:
                with open("../data/train.ori", 'r') as vf:
                    V, D = kf["V"][...], kf["D"][...]
                    # 研究轨迹1的中间点值
                    vf = vf.readlines()
                    labels = labelF.readlines()
                    traOfSearch = 1

                    # stream = vf[(traOfSearch - 1) * 3200 + 131]
                    # stream = vf[6403]
                    stream = vf[6400]
                    s = [int(x) for x in stream.split()]
                    point = len(s) // 2
                    myvalue = s[point]
                    neighborvalues = V[s[point]]
                    for i in range(0, 20799, 400):#, 1600):
                        # locations = f['trips'][str((traOfSearch - 1) * 3200 + i+1)]
                        # timestamps = f['timestamps'][str((traOfSearch - 1) * 3200 + i+1)]
                        locations = f['trips'][str(i+1)]
                        trip = []
                        for (lon, lat) in locations:
                            trip.append([lon, lat])
                        seq = np.array(trip2seq(region, trip))
                        # label = f['labels'][str((traOfSearch - 1) * 3200 + i+1)]
                        # label = labels[(traOfSearch - 1) * 3200 + i]
                        label = labels[i]
                        label = label.split(",")
                        ori_label = int(label[1]) #+ 1
                        pre_label = int(label[2])
                        # pre_label = ori_label
                        # if (not pre_label == 29) and (not ori_label == 8) and (not ori_label == 12):
                        #     continue
                        # if (not ori_label == 12):
                        #     continue
                        for idx, (lon, lat) in enumerate(trip):
                            cell = gps2cell(region, lon, lat)
                            x, y = cell2coord(region, cell)
                            trip[idx] = [x, y]
                        trip = np.array(trip)
                        # if (not ori_label == 35) and (not pre_label == 1) and (not ori_label == 8) and (not ori_label == 44):
                        #     continue
                        # stream = vf[i]
                        # s = np.array([int(x) for x in stream.split()])
                        ismine = seq == myvalue
                        neighborvalue = np.array([True if x in neighborvalues else False for x in seq])
                        # ismine = np.ones_like(ismine)
                        # neighborvalue = np.ones_like(neighborvalue)
                        # if pre_label != 1:
                        #     continue
                        # if ~(np.array(ismine).any()):
                        #     continue
                        # if ~(neighborvalue.any() or np.array(ismine).any()):
                        #     continue
                        # fig = plt.figure()
                        # ax = Axes3D(fig)

                        # ax.plot3D(locations[:,:1].reshape(-1, ), locations[:,1:2].reshape(-1, ), timestamps[:].reshape(-1, ), c = colorsOfLabel[pre_label], label = str(ori_label) + "," + str(pre_label))
                        # ax.scatter3D(locations[0, :1], locations[0, 1:2], timestamps[0], c='r', marker='o')
                        # ax.scatter3D(locations[-1, :1], locations[-1, 1:2], timestamps[-1], c='g', marker='o')
                        ax.plot(trip[:,:1].reshape(-1, ), trip[:,1:2].reshape(-1, ), c = colorsOfLabel[pre_label], label = str(ori_label) + "," + str(pre_label))
                        # ax.scatter3D(locations[:,:1][np.logical_and(~ismine, ~neighborvalue)], locations[:,1:2][np.logical_and(~ismine, ~neighborvalue)], timestamps[:][np.logical_and(~ismine, ~neighborvalue)], c='b', marker='o')
                        ax.scatter(trip[:,:1][neighborvalue], trip[:,1:2][neighborvalue], c='g', marker='o')
                        ax.scatter(trip[:,:1][ismine], trip[:,1:2][ismine], c='r', marker='o')

                        # ax.set_xlabel('lon')  # 画出坐标轴
                        # ax.set_ylabel('lat')
                        # ax.set_zlabel('timestamp')
                        # plt.show()

                    ax.set_xlabel('lon') # 画出坐标轴
                    ax.set_ylabel('lat')
                    handles, labels = ax.get_legend_handles_labels()
                    handle_list, label_list = [], []
                    for handle, label in zip(handles, labels):
                        if label not in label_list:
                            handle_list.append(handle)
                            label_list.append(label)
                    plt.legend(handle_list, label_list)
                    plt.show()