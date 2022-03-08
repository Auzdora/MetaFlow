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
from utils import euclidean_distance
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


if __name__ == "__main__":
    data_set = [[0,0],[0,1],[1,1],[4,4],[5,5]]
    kmeans = Kmeans(data_set, 5)
    kmeans.train()