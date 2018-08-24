# -*- coding: utf-8 -*-

from numpy import *


def load_data_set(filename):
    data_mat = []
    fr = open(filename)
    for line in fr.readlines():
        curline = line.strip().split('\t')
        curline = map(float, curline)
        data_mat.append(curline)
    return data_mat


def dist_eclud(veca, vecb):
    return sqrt(sum(power(veca-vecb, 2)))


def rand_cent(dataset, k):
    n = shape(dataset)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minj = min(dataset[:, j])
        rangej = float(max(dataset[:, j]) - minj)
        centroids[:, j] = minj + rangej * random.rand(k, 1)
    return centroids


def kmeans(dataset, k, dist_meas=dist_eclud, create_cent=rand_cent):
    m = shape(dataset)[0]
    cluster_assment = mat(zeros((m, 2)))
    centroids = create_cent(dataset, k)
    changed = True
    while changed:
        changed = False
        for i in range(m):
            min_dist = inf; min_index = -1
            for j in range(k):
                dist = dist_meas(dataset[i, :], centroids[j, :])
                if dist < min_dist:
                    min_dist = dist
                    min_index = j
            if cluster_assment[i, 0] != min_index:
                changed = True
            cluster_assment[i, :] = min_index, min_dist**2
        print(centroids)
        for i in range(k):
            cluster = dataset[nonzero(cluster_assment[:, 0].A == i)[0]]
            # cluster = []
            # for j in range(m):
            #     if cluster_assment[j, 0] == i:
            #         cluster.append(dataset[j])
            centroids[i, :] = mean(cluster, axis=0)
    return centroids, cluster_assment
