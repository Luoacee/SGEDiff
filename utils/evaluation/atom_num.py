"""Utils for sampling size of a molecule of a given protein pocket."""

import numpy as np
from scipy import spatial as sc_spatial

from utils.evaluation.atom_num_config import CONFIG


def get_space_size(pocket_3d_pos):  # 传入一堆坐标，计算这些坐标的两两坐标差，排序后，选择top10坐标的中位数作为蛋白的空间尺度
    aa_dist = sc_spatial.distance.pdist(pocket_3d_pos, metric='euclidean')
    aa_dist = np.sort(aa_dist)[::-1]
    return np.median(aa_dist[:10])


def _get_bin_idx(space_size):
    bounds = CONFIG['bounds']  # 载入初始化范围
    for i in range(len(bounds)):  # 遍历每一个盒子大小范围
        if bounds[i] > space_size:  # 如果盒子大小合适，则返回idx
            return i
    return len(bounds)


def sample_atom_num(space_size): # 根据蛋白口袋的空间尺度采样配体原子
    bin_idx = _get_bin_idx(space_size) # 得到合适的盒子idx
    num_atom_list, prob_list = CONFIG['bins'][bin_idx]  # 根据盒子大小选择生成的原子数量和对应的概率
    return np.random.choice(num_atom_list, p=prob_list) # 根据概率选择生成的原子个数
