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
        # print(centroids)
        for i in range(k):
            cluster = dataset[nonzero(cluster_assment[:, 0].A == i)[0]]
            # cluster = []
            # for j in range(m):
            #     if cluster_assment[j, 0] == i:
            #         cluster.append(dataset[j])
            centroids[i, :] = mean(cluster, axis=0)
    return centroids, cluster_assment


def bikmeans(dataset, k, dist_meas=dist_eclud):
    m = shape(dataset)[0]
    cluster_assment = mat(zeros((m, 2)))
    # start with all the points in one cluster
    centroid0 = mean(dataset, axis=0).tolist()[0]
    cents = [centroid0]
    for j in range(m):
        cluster_assment[j, 1] = dist_meas(mat(centroid0), dataset[j, :])**2
    best_cent = 0
    best_new_cents = mat([])
    best_new_cluster_assment = mat([])
    while len(cents) < k:
        lowest_sse = inf
        for i in range(len(cents)):
            pts_in_curr_cluster = dataset[nonzero(cluster_assment[:, 0].A == i)[0], :]
            new_cents, new_cluster_assment = kmeans(pts_in_curr_cluster, 2, dist_meas)
            split_sse = sum(new_cluster_assment[:, 1])
            rest_sse = sum(cluster_assment[nonzero(cluster_assment[:, 0].A != i)[0], 1])
            print "split sse and rest see: ", split_sse, rest_sse
            if (split_sse + rest_sse) < lowest_sse:
                best_cent = i
                best_new_cents = new_cents
                best_new_cluster_assment = new_cluster_assment.copy()
                lowest_sse = split_sse + rest_sse
        best_new_cluster_assment[nonzero(best_new_cluster_assment[:, 0].A == 1)[0], 0] = len(cents)
        best_new_cluster_assment[nonzero(best_new_cluster_assment[:, 0].A == 0)[0], 0] = best_cent
        print 'the best_cent is: ', best_cent
        print 'the len of best_new_cluster_assment is: ', len(best_new_cluster_assment)
        cents[best_cent] = best_new_cents[0, :].tolist()[0]
        cents.append(best_new_cents[1, :].tolist()[0])
        # update cluster_assment with best_new_cluster_assment
        cluster_assment[nonzero(cluster_assment[:, 0].A == best_cent)[0]] = best_new_cluster_assment
        # curr_cluster_point_index = []
        # for i in range(m):
        #     if cluster_assment[i, 0] == best_cent:
        #         curr_cluster_point_index.append(i)
        # for j in range(len(best_new_cluster_assment)):
        #     cluster_assment[curr_cluster_point_index[j], :] = best_new_cluster_assment[j, :]
    return mat(cents), cluster_assment
