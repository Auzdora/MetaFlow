"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved

    Filename: cluster.py
    Description: This file provide multiple clustering algorithm.

    Created by Melrose-Lbt 2022-3-7
"""
import random
import numpy as np
import copy
from utils import euclidean_distance, find_min_dis
from core import Tensor


class Kmeans:
    """
        K-means algorithm.
    """
    def __init__(self, dataset, k):
        self.dataset = dataset
        self.k = k
        self.cluster_center = self.random_choose()
        self.dot_set = [[] for i in range(self.k)]
        self.last_dot_set = [[] for i in range(self.k)]

    def init_cluster(self):
        """
            Initialize dot set.
        :return:
        """
        for i in range(self.k):
            self.dot_set[i].append(self.cluster_center[i])

    def random_choose(self):
        """
            Choose random dots as initial dots from dataset.
        :return:
        """
        cluster_center = []
        cluster_index = random.sample(range(0, len(data_set)), self.k)
        for i in range(self.k):
            cluster_center.append(self.dataset[cluster_index[i]])
        print(cluster_center)

        return cluster_center

    def center_stable(self):
        """
            Judge center point is stable or not.
        :return: bool value
        """
        null = 0
        for i in range(self.k):
            if len(self.last_dot_set[i]) == 0:
                null += 1
        if null == self.k:
            return True

        same_cnt = 0
        for i in range(self.k):
            ith_cluster = np.array(self.dot_set[i])
            last_ith_cluster = np.array(self.last_dot_set[i])
            # First check shape, because numpy needs same shape matrix to compare
            if ith_cluster.shape == last_ith_cluster.shape:
                # If they have same shape, then check their elements
                if (ith_cluster == last_ith_cluster).all():
                    same_cnt += 1
                else:
                    continue
            else:
                continue

        if same_cnt == self.k:
            return False
        else:
            return True

    def check_dot_exist(self, dot):
        """
            Called by refresh_dot_set function. Check if this dot exists in
        current dot set.
        :param dot: a single dot
        :return: bool, index
        """
        for cluster_index in range(self.k):
            if dot in self.dot_set[cluster_index]:
                return True, cluster_index
        return False, -1

    def refresh_dot_set(self, min_index, dot):
        """
            Refresh dot set to the renewed one.
        :param min_index: minimum distance index
        :param dot: a single dot
        """
        bool_value, cluster_index = self.check_dot_exist(dot)
        if bool_value:
            if cluster_index == min_index:
                pass
            else:
                self.dot_set[cluster_index].remove(dot)
                self.dot_set[min_index].append(dot)
        else:
            self.dot_set[min_index].append(dot)

    def compute_centers(self):
        """
            Compute cluster's center value.
        """
        self.cluster_center = []
        for cluster_index in range(self.k):
            new_center = np.mean(np.array(self.dot_set[cluster_index]),axis=0)
            self.cluster_center.append(list(new_center))

    def train(self):
        """
            Train logic.
        """
        # Choose k dots as initial cluster centers
        self.init_cluster()

        while self.center_stable():
            self.last_dot_set = copy.deepcopy(self.dot_set)
            for items in self.dataset:
                dis_set = []
                for i in range(self.k):
                    dis_set.append(euclidean_distance(items, self.cluster_center[i]))
                min_dis_index = dis_set.index(min(dis_set))
                self.refresh_dot_set(min_dis_index, items)
            self.compute_centers()

        print(self.dot_set)


class HierarchicalClustering:
    """
        Hierarchical clustering algorithm.
    """
    def __init__(self, dataset, threshold, method='min'):
        """

        :param dataset: dataset
        :param threshold: if you choose k, it'll stop at kth cluster
        :param method: 'min': single linkage
                       'max': complete linkage
                       'average': average linkage
        """
        self.dataset = np.array(dataset)
        self.threshold = threshold
        self.method = method
        self.data_num = np.array(dataset).shape[0]
        self.dot_set = [[] for i in range(self.data_num)]

    def init_dot_set(self):
        for i, cluster in enumerate(self.dot_set):
            cluster.append(list(self.dataset[i]))

    def single_linkage(self):
        self.init_dot_set()
        dis_matrix = np.zeros((self.data_num, self.data_num))
        for index in range(self.data_num):
            for sub_index in range(index, self.data_num):
                dis_matrix[index][sub_index] = find_min_dis(self.dot_set[index], self.dot_set[sub_index])
        # dis_matrix += dis_matrix.T - np.diag(dis_matrix.diagonal())

        row_index, col_index = np.where(dis_matrix == np.min(dis_matrix[np.nonzero(dis_matrix)]))
        print(dis_matrix)
        print(np.nonzero(dis_matrix))
        print(row_index, col_index)

        while len(self.dot_set) > self.threshold:
            row_index, col_index = np.where(dis_matrix == np.min(dis_matrix[np.nonzero(dis_matrix)]))

            if row_index.shape[0] == 1:
                self.dot_set[row_index[0]] = self.dot_set[row_index[0]] + self.dot_set[col_index[0]]
                self.dot_set.remove(self.dot_set[col_index[0]])
            elif row_index.shape[0] > 1:
                concat_index = np.unique(np.array(list(row_index) + list(col_index)))
                del_list = []
                # register del list
                for index in concat_index:
                    del_list.append(self.dot_set[index])

                for items in concat_index:
                    if items == concat_index[0]:
                        continue
                    self.dot_set[concat_index[0]] += self.dot_set[concat_index[items]]

                for i in range(len(del_list)):
                    if i == 0:
                        continue
                    self.dot_set.remove(del_list[i])

            # renew distance matrix
            dis_matrix = np.zeros((len(self.dot_set), len(self.dot_set)))
            for index in range(len(self.dot_set)):
                for sub_index in range(index, len(self.dot_set)):
                    dis_matrix[index][sub_index] = find_min_dis(self.dot_set[index], self.dot_set[sub_index])


        print(self.dot_set)

    def complete_linkage(self):
        pass

    def average_linkage(self):
        pass

    def train(self):
        """
            train logic
        """
        if self.method == 'min':
            self.single_linkage()
        elif self.method == 'max':
            self.complete_linkage()
        elif self.method == 'average':
            self.average_linkage()


if __name__ == "__main__":
    data_set = [[0,0, 0],[0,1, 1],[1,1, 1],[4,4, 6],[5,5, 2]]
    cluster = HierarchicalClustering(data_set, 1, 'min')
    cluster.train()