"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: distance.py
    Description: This file contains distance calculation methods.

    Created by Melrose-Lbt 2022-3-7
"""
import numpy as np
from numpy import ndarray
from core import Tensor
import math


def euclidean_distance(tensor1, tensor2):
    assert type(tensor1) == type(tensor2), "tensor1 type:{} is not same as tensor2 type:{}"\
        .format(type(tensor1), type(tensor2))

    if isinstance(tensor1, ndarray):
        assert tensor1.shape == tensor2.shape, "tensor1 shape:{} is not aligned with tensor2 shape:{}"\
            .format(tensor1.shape, tensor2.shape)
        return np.sqrt(((tensor1 - tensor2)**2).sum())

    elif isinstance(tensor1, Tensor):
        assert tensor1.shape == tensor2.shape, "tensor1 shape:{} is not aligned with tensor2 shape:{}" \
            .format(tensor1.shape, tensor2.shape)
        return Tensor(np.sqrt(((tensor1.value - tensor2.value)**2).sum()))

    elif isinstance(tensor1, list):
        assert len(tensor1) == len(tensor2), "tensor1 shape:{} is not aligned with tensor2 shape:{}" \
            .format(len(tensor1), len(tensor2))
        return np.sqrt(((np.array(tensor1) - np.array(tensor2))**2).sum())

    else:
        origin_type = type(tensor1)
        return origin_type(np.sqrt(((tensor1 - tensor2)**2)))


def find_min_dis(cluster1, cluster2):
    """
                Cluster1 and cluster2 are two vectors. Then they'll be transformed to matrix
            to speed up computation process.
            :param cluster1: a vector
            :param cluster2: a vector
            :return: min distance between two clusters
            """
    assert type(cluster1) == type(cluster2), "cluster1 type:{} and cluster2 type:{} are not same." \
        .format(type(cluster1), type(cluster2))

    if isinstance(cluster1, list):
        cluster1 = np.array(cluster1)
        cluster2 = np.array(cluster2)
    # TODO: dimension in one dataset maybe different, this is wrong, assert needs to be added in the future
    dim_of_dot = len(cluster1[0])

    mat_cluster1 = np.tile(cluster1.flatten(), (cluster2.shape[0], 1)).T
    mat_cluster2 = np.tile(cluster2.T, (cluster1.shape[0], 1))

    dis = (mat_cluster1 - mat_cluster2)**2
    real_dis = []
    for items in range(int(dis.shape[0]/dim_of_dot)):
        add_dis = np.sqrt(np.sum(dis[dim_of_dot*items:dim_of_dot*items+dim_of_dot, :], axis=0))
        real_dis.append(add_dis)
    real_dis = np.array(real_dis)
    return np.min(real_dis)


def find_max_dis(cluster1, cluster2):
    """
                Cluster1 and cluster2 are two vectors. Then they'll be transformed to matrix
            to speed up computation process.
            :param cluster1: a vector
            :param cluster2: a vector
            :return: min distance between two clusters
            """
    assert type(cluster1) == type(cluster2), "cluster1 type:{} and cluster2 type:{} are not same." \
        .format(type(cluster1), type(cluster2))

    if isinstance(cluster1, list):
        cluster1 = np.array(cluster1)
        cluster2 = np.array(cluster2)
    # TODO: dimension in one dataset maybe different, this is wrong, assert needs to be added in the future
    dim_of_dot = len(cluster1[0])

    mat_cluster1 = np.tile(cluster1.flatten(), (cluster2.shape[0], 1)).T
    mat_cluster2 = np.tile(cluster2.T, (cluster1.shape[0], 1))

    dis = (mat_cluster1 - mat_cluster2)**2
    real_dis = []
    for items in range(int(dis.shape[0]/dim_of_dot)):
        add_dis = np.sum(dis[dim_of_dot*items:dim_of_dot*items+dim_of_dot, :], axis=0)
        real_dis.append(add_dis)
    real_dis = np.array(real_dis)
    print(real_dis)
    return np.max(real_dis)


if __name__ =="__main__":
    a = np.array([[1,1, 1], [0, 0, 1], [2,3, 2], [4,7,1]])
    b = np.array([[2,1,1], [0, 0,2], [1,1,1]])
    print(find_min_dis(a, b))
