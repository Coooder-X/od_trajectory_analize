import os
import pickle, json
from SpatialRegionTools import SpatialRegion, makeKneighbor, createTrainVal, createTrainVal_OnlyOriginal, \
    saveKNearestVocabs, makeVocab
from utils import downsamplingDistort
# from spatio_split import dbscan, stme
import datetime, time


def dir_delete(dir_name):
    files = os.listdir(dir_name)
    # 遍历删除指定目录下的文件
    for file in files:
        os.remove(os.path.join(dir_name, file))
        print(file, "删除成功")
    print(dir_name, "删除成功")


k = 20  # 20
kt = 10  # 10
min_pts = 14  # 13

if __name__ == "__main__":
    # 时间计算
    start_dt = datetime.datetime.now()
    start_t = time.time()
    print("START DATETIME")
    print(start_dt)

    ## 1, 初始化
    with open("../conf/preprocess_conf.json") as conf:
        # 读取配置文件
        json_data = json.load(conf)
        needTime = json_data["needTime"] == 1
        timecellsize, interesttime, nointeresttime, delta, timefuzzysize, timestart = 1, 1.0, 1.0, 1.0, 1.0, 1.0  # 时间距离公式的参数 todo：待定
        if needTime:
            timecellsize = json_data["timecellsize"]  # 时间上的size
            interesttime = json_data["interesttime"]  # 计算k近邻时候的时间阈值
            nointeresttime = json_data["nointeresttime"]  # 计算k近邻时候的时间阈值
            delta = json_data["delta"]  # 计算k近邻时候的时间阈值
            timefuzzysize = json_data["timefuzzysize"]
            timestart = json_data["timestart"]
            assert timecellsize >= timefuzzysize * 2
        cellsize = json_data["distcellsize"]
        cityname = json_data["cityname"]
        paramfile = json_data["paramfile"]  # 存放研究区域的参数
        stme_file = json_data["spatiosplitobj"] + "_" + str(k) + "_" + str(kt) + "_" + str(min_pts) + ".pkl"
        use_grid = json_data["usegrid"] == 1
        has_label = json_data["hasLabel"] == 1
    # 删除上一次的结果
    # dir_delete("../data/")

    # 读取该目录所有文件 也就是需要解析的所有文件
    h5_files = os.listdir(json_data["h5dirname"])
    # 完善文件路径
    for i in range(len(h5_files)):
        h5_files[i] = os.path.join(json_data["h5dirname"], h5_files[i])
        print(h5_files[i])

    print("是否考虑时间: " + str(needTime))

    # points, spatio_num, spatio_pos = dbscan(h5_files)
    # hulls = stme(h5_files)

    # with open(stme_file, 'rb') as f:  # open file with write-mode
    #     hulls = pickle.loads(f.read())
    hulls = []

    # 研究区域和实验配置类
    region = SpatialRegion(cityname,
                           # 120.170, 30.232, # original 原来180
                           # 120.047, 30.228,  # original 原来180
                           # 120.098, 30.263,
                           # 120.197, 30.320,
                           119.398, 30.263, # 整个hz
                           120.797, 30.320, # 整个hz
                           0, 86400,  # 时间范围,一天最大86400(以0点为相对值)
                           cellsize, cellsize,
                           timecellsize,  # 时间步
                           1,  # minfreq 最小击中次数
                           40_0000,  # maxvocab_size
                           30,  # k
                           4,  # vocab_start 词汇表基准值
                           interesttime,  # 时间阈值
                           nointeresttime,
                           delta,
                           needTime,
                           2, 4000,
                           timefuzzysize, timestart,
                           hulls, use_grid, has_label)
    # points, spatio_num, spatio_pos)

    ## 2, 处理数据
    print("Creating paramter file $paramfile")
    train_datas, train_labels, val_datas, val_labels, test_datas, test_labels = makeVocab(region, h5_files)

    # todo
    # makeKneighbor(region)
    createTrainVal(region, train_datas, False, downsamplingDistort)  # 生成train.src,train.trg,val.src,val.trg
    createTrainVal(region, val_datas, True, downsamplingDistort)  # 生成train.src,train.trg,val.src,val.trg
    # 生成原轨迹的训练集文件和验证集文件，不生成带噪声的训练集文件和验证集文件
    createTrainVal_OnlyOriginal(region, train_datas, train_labels, False, False)  # 生成train.ori，val.ori
    createTrainVal_OnlyOriginal(region, val_datas, val_labels, True, False)  # 生成train.ori，val.ori
    createTrainVal_OnlyOriginal(region, test_datas, test_labels, False, True)  # 生成train.ori，val.ori
    print("createTrainVal() finished!")
    saveKNearestVocabs(region)

    # 存下区域信息
    output_hal = open("../data/region.pkl", 'wb')
    region_save = pickle.dumps(region)
    output_hal.write(region_save)
    output_hal.close()
    print("Vocabulary size %d with dist_cell size %d (meters) and time_cell size %d" % (
    region.vocab_size, cellsize, timecellsize))

    end_dt = datetime.datetime.now()
    end_t = time.time()
    print("END DATETIME")
    print(end_dt)
    print("Total time: " + str(end_t - start_t))
