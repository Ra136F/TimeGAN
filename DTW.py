import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def compute_dtw_distance(sequence_1, sequence_2):
    """
    计算两个多维时间序列之间的DTW距离，支持不同长度的序列。

    参数:
    - sequence_1: 第一个时间序列，形状为 (T1, D)，其中 T1 是时间步长，D 是特征维度
    - sequence_2: 第二个时间序列，形状为 (T2, D)，其中 T2 是时间步长，D 是特征维度

    返回:
    - dtw_distance: DTW距离
    - alignment_path: 最优对齐路径
    """

    # 将多维时间序列数据转换为适合DTW的格式 [(time, [feature_1, feature_2, ...])]
    # seq1 = [(i, list(sequence_1[i])) for i in range(len(sequence_1))]
    # seq2 = [(i, list(sequence_2[i])) for i in range(len(sequence_2))]
    seq1=np.array(sequence_1)
    seq2=np.array(sequence_2)

    # 计算DTW距离
    dtw_distance, alignment_path = fastdtw(seq1, seq2, dist=euclidean)

    return dtw_distance, alignment_path



import numpy as np


def calculate_rmse(y_true, y_pred):
    """
    计算两个序列的 RMSE (Root Mean Square Error)

    参数：
    y_true -- 真实值序列
    y_pred -- 预测值序列

    返回：
    RMSE值
    """
    if len(y_true) != len(y_pred):
        raise ValueError("两个序列的长度必须相同")

    # 计算误差平方和
    squared_errors = [(y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true))]

    # 计算均方根误差
    rmse = np.sqrt(np.mean(squared_errors))
    return rmse






