import random, math

import numpy as np

from data_process.SpatialRegionTools import inregionT, inregionS


def lonlat2meters(lon, lat):
    semimajoraxis = 6378137.0
    east = lon * 0.017453292519943295
    north = lat * 0.017453292519943295
    t = math.sin(north)
    return semimajoraxis * east, 3189068.5 * math.log((1 + t) / (1 - t))

def meters2lonlat(x, y):
    semimajoraxis = 6378137.0
    lon = x / semimajoraxis / 0.017453292519943295
    t = math.exp(y / 3189068.5)
    lat = math.asin((t - 1) / (t + 1)) / 0.017453292519943295
    return lon, lat

def downsampling(trj_data, dropping_rate):
    noisetrip = []
    noisetrip.append(trj_data[0])
    for i in range(1, len(trj_data)-1):
        if random.random() > dropping_rate:
            noisetrip.append(trj_data[i])
    noisetrip.append(trj_data[len(trj_data)-1])
    return noisetrip

def distort(region, trip, distorting_rate, radius):
    noisetrip = []
    if region.needTime:
        for (i, [lon, lat, time]) in enumerate(trip):
            if random.random() <= distorting_rate:
                distort_dist = random.randint(0, radius)
                x, y = lonlat2meters(lon, lat)
                xnoise, ynoise = random.uniform(0,2) - 1, random.uniform(0,2) - 1
                normz = math.hypot(xnoise, ynoise)
                xnoise, ynoise = xnoise * distort_dist/normz, ynoise * distort_dist/normz
                nlon, nlat = meters2lonlat(x + xnoise, y + ynoise)
                if not (inregionT(region, nlon, nlat, time)):
                    nlon = lon
                    nlat = lat
                noisetrip.append([nlon, nlat, time])
            else:
                noisetrip.append(trip[i])
    else:
        for (i, [lon, lat]) in enumerate(trip):
            if random.random() <= distorting_rate:
                distort_dist = random.randint(0, radius)
                x, y = lonlat2meters(lon, lat)
                xnoise, ynoise = random.uniform(0,2) - 1, random.uniform(0,2) - 1
                normz = math.hypot(xnoise, ynoise)
                xnoise, ynoise = xnoise * distort_dist/normz, ynoise * distort_dist/normz
                nlon, nlat = meters2lonlat(x + xnoise, y + ynoise)
                if not (inregionS(region, nlon, nlat)):
                    nlon = lon
                    nlat = lat
                noisetrip.append([nlon, nlat])
            else:
                noisetrip.append(trip[i])
    return noisetrip

def distort_time(region, trip, distorting_time):
    noisetrip = []
    seconds = random.randint(-distorting_time, distorting_time)

    for (lon, lat, time) in trip:        
        offsettime = time + seconds
        if offsettime >= 86400:
            offsettime = offsettime - 86400
        if offsettime < 0:
            offsettime = offsettime + 86400
        if not inregionT(region, lon, lat, time):
            return trip
        noisetrip.append([lon, lat, offsettime])
    return noisetrip

def downsamplingDistort(trj_data, region):
    noisetrips = []
    dropping_rates = [0, 0.2, 0.4, 0.6]
    # dropping_rates = [0, 0.05, 0.1, 0.15, 0.2]
    distorting_rates = [0, 0.2, 0.4, 0.6]
    distort_radius = 30.0
    distorting_time = 900
    for dropping_rate in dropping_rates:
        noisetrip1 = downsampling(trj_data, dropping_rate)
        if not (region.min_length <= len(noisetrip1) <= region.max_length):
            noisetrip1 = trj_data
        for distorting_rate in distorting_rates:
            noisetrip2 = distort(region, noisetrip1, distorting_rate, distort_radius)
            if region.needTime:
                # for time in range(20):
                noisetrip3 = distort_time(region, noisetrip2, distorting_time)
                noisetrips.append(noisetrip3)
            else:
                noisetrips.append(noisetrip2)
    return noisetrips


def cal_meter_dist(coord1, coord2):
    x1, y1 = coord1[0], coord1[1]
    x2, y2 = coord2[0], coord2[1]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


class UnionFindSet(object):
    """并查集"""
    def __init__(self, data_list):
        """初始化两个字典，一个保存节点的父节点，另外一个保存父节点的大小
        初始化的时候，将节点的父节点设为自身，size设为1"""
        self.father_dict = np.array([i for i in range(len(data_list))])
        self.size_dict = np.array([1 for i in range(len(data_list))])

        # for node in data_list:
        #     self.father_dict[node] = node
        #     self.size_dict[node] = 1

    def find_head(self, node):
        """使用递归的方式来查找父节点

        在查找父节点的时候，顺便把当前节点移动到父节点上面
        这个操作算是一个优化
        """
        father = self.father_dict[node]
        while father != self.father_dict[father]:
            father = self.father_dict[father]
        # if node != father:
        #     father = self.find_head(father)
        self.father_dict[node] = father
        return father

    def get_size(self, node):
        return self.size_dict[self.find_head(node)]

    def is_same_set(self, node_a, node_b):
        """查看两个节点是不是在一个集合里面"""
        return self.find_head(node_a) == self.find_head(node_b)

    def union(self, node_a, node_b):
        """将两个集合合并在一起"""
        if node_a is None or node_b is None:
            return

        a_head = self.find_head(node_a)
        b_head = self.find_head(node_b)

        if a_head != b_head:
            a_set_size = self.size_dict[a_head]
            b_set_size = self.size_dict[b_head]
            self.father_dict[b_head] = a_head
            self.size_dict[a_head] = a_set_size + b_set_size

if __name__ == '__main__':
    a = [1,2,3,4,5]
    union_find_set = UnionFindSet(a)
    union_find_set.union(1,2)
    union_find_set.union(3,5)
    union_find_set.union(3,1)
    print(union_find_set.is_same_set(2,5))  # True