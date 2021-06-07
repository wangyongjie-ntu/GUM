#Filename:	submodular.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sel 17 Des 2019 06:40:47  WIB

import numpy as np
import itertools
import time
import heapq

def CELF(data, K = 2, C = 3):
    # data: a collection of records
    # K: top K attributes
    # C: cluster numbers
    start = time.time()
    length, width = data.shape
    SubsetC = []
    partitions = [[] for _ in range(C)]
    attributes_sets = list(itertools.combinations(range(width), K))
    # add the first K-attribute set
    column_sum = np.sum(data, axis = 0)
    column_sort = np.argsort(column_sum, kind = "stable")
    SubsetC.append(tuple(np.sort(column_sort[-K:]).tolist()))
    # remove the first one from all combinations
    attributes_sets.remove(SubsetC[0])
     
    largest_sum = np.sum(np.sort(column_sum, kind = "stable")[-K:])
    # add the second K-attribute set
    heap = []
    if len(SubsetC) < C:
        for i, current_set in enumerate(attributes_sets):
            tmp = np.take(data, [current_set, SubsetC[0]], axis = 1)
            tmp_sum = np.zeros((length, 2))
            tmp_sum[:,0] = tmp[:, 0, :].sum(1)
            tmp_sum[:,1] = tmp[:, 1, :].sum(1)
            maximum = np.max(tmp_sum, axis = 1)
            heapq.heappush(heap, (-(np.sum(maximum) - largest_sum), current_set))

        # initialize subsetC as empty set
        #print(heap)
        topx = heapq.heappop(heap)
        SubsetC.append(topx[1])
        largest_sum -= topx[0]

    while len(SubsetC) < C:
        _, current_set = heapq.heappop(heap)
        tmp_sum = np.zeros((length, len(SubsetC) + 1))
        for m in range(len(SubsetC)):
            selected_attr = data[:, SubsetC[m]]
            tmp_sum[:, m] = np.sum(selected_attr, axis = 1)

        tmp_sum[:, -1] = np.sum(data[:, current_set], axis = 1)
        maximum = np.max(tmp_sum, axis = 1)
        maximum = sum(maximum)
        heapq.heappush(heap, (-(maximum - largest_sum), current_set))
        topx = heapq.nsmallest(1, heap)[0]
        if topx[1] == current_set:
            #print(heap)
            SubsetC.append(current_set)
            largest_sum = maximum
            heapq.heappop(heap)

    #obtain the partitions under subsetC
    tmp_sum = np.zeros((length, len(SubsetC)))
    for m in range(len(SubsetC)):
        selected_attr = data[:, SubsetC[m]]
        tmp_sum[:, m] = np.sum(selected_attr, axis = 1)

    tmp = np.argmax(tmp_sum, 1)
    for i in range(C):
        partitions[i] = np.where(tmp == i)[0].tolist()
          
    end = time.time()

    return end - start, largest_sum, SubsetC, partitions
