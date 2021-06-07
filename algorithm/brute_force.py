import numpy as np
import itertools
import time

def brute_force_algorthm(data, L = 2, K =  3):

    '''
    data: a collection of records
    K: top K attributes
    C: cluster numbers
    '''
    start = time.time()
    length, width = data.shape
    attributes_sets = itertools.combinations(range(width), L)
    All_possible_clusters = itertools.combinations(attributes_sets, K)
    SubsetC = None
    partitions = None
    largest_sum = 0

    for _, current_cluster_attribute in enumerate(All_possible_clusters):

        current_cluster_partition = [[] for _ in range(K)] 
        cluster_sum = np.zeros((length, K))
        for j, attribute in enumerate(current_cluster_attribute):
            cluster_sum[:, j] = np.sum(np.take(data, attribute, axis = 1), axis = 1)

        tmp = np.argmax(cluster_sum, axis = 1)
        for j in range(K):
            current_cluster_partition[j] = np.where(tmp == j)[0].tolist()
        
        total_sum = np.sum(np.max(cluster_sum, axis = 1))
        if largest_sum < total_sum:
            largest_sum = total_sum
            SubsetC = current_cluster_attribute
            partitions = current_cluster_partition

    end = time.time()
    time_elapsed = end - start
    return time_elapsed, largest_sum, SubsetC, partitions

if __name__ == "__main__":
    
    data = np.array([[2, 0, 3, 3, 2], [4, 1, 4, 3, 1], [2, 4, 4, 0, 1], [3, 2, 1, 0, 1], [2, 4, 4, 3, 1], [3, 2, 3, 3, 3]])
    time_elapsed, largest_sum, SubsetC, partitions = brute_force_algorthm(data)
    print(largest_sum)
    print(SubsetC)
    print(partitions)

