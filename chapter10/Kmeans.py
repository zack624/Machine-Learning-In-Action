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


# spherical law of cosines
def dist_SLC(veca, vecb):
    a = sin(veca[0, 1]*pi/180) * sin(vecb[0, 1]*pi/180)
    b = cos(veca[0, 1]*pi/180) * cos(vecb[0, 1]*pi/180) * cos(pi*(vecb[0, 0] - veca[0, 0])/180)
    return arccos(a + b) * 6371.0


# apply bikmeans and plot all the points
import matplotlib
import matplotlib.pyplot as plt
def cluster_clubs(num_cluster=5):
    # load data of longitude and latitude from places.txt
    dat_list = []
    fr = open('places.txt')
    for line in fr.readlines():
        cur_line = line.split('\t')
        dat_list.append([float(cur_line[4]), float(cur_line[3])])
    dat_mat = mat(dat_list)
    centroids, cluster_assment = bikmeans(dat_mat, num_cluster, dist_meas=dist_SLC)
    # plot points and centroids
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8] # left,bottom,width,height
    scatter_markers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    # load Portland.png
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgp = plt.imread('Portland.png')
    ax0.imshow(imgp)
    # plot points of every cluster and centroids
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(num_cluster):
        pts_in_curr_cluster = dat_mat[nonzero(cluster_assment[:, 0].A==i)[0], :]
        marker_style = scatter_markers[i % len(scatter_markers)]
        ax1.scatter(pts_in_curr_cluster[:, 0].flatten().A[0],\
                    pts_in_curr_cluster[:, 1].flatten().A[0], marker=marker_style, s=90)
    ax1.scatter(centroids[:, 0].flatten().A[0],\
                centroids[:, 1].flatten().A[0], marker='+', s=300)
    plt.show()
