import numpy as np
import time
import random

def random_sampling(data, init_centriod, L, K):
    '''
    data: a collection of records
    init_centriod: the inital centriod
    l: l attributes
    k: cluster numbers
    '''
    start = time.time()
    length, width = data.shape
    partitions = [[] for _ in range(K)]
    attribute_sum = np.zeros((K, width))
    largest_sum = 0
    
    tmp_sum = np.zeros((length, len(init_centriod)))
    selected_data = np.take(data, init_centriod, axis = 1)
    for k in range(K):
        tmp_sum[:, k] = selected_data[:, k, :].sum(1)
    
    tmp = np.argmax(tmp_sum, 1)
    largest_sum = np.sum(np.max(tmp_sum, 1))
    for k in range(K):
        partitions[k] = np.where(tmp == k)[0].tolist()

    end = time.time()
    return end - start, largest_sum, init_centriod, partitions

def multiple_random_sampling(data, L = 2, K = 3, init_num = 1):
    '''
    data: a collection of records
    init_centriod: the inital centriod
    K: top K attributes
    C: cluster numbers
    init_num: the number of different initialization
    '''
    # randomly initalize the centriod of each cluster
    length, width = data.shape
    func = lambda x : random.sample(range(width), x)
    init_centriod_sets = [[] for _ in range(init_num)]
    
    # generate different initialization
    for i in range(init_num):
        while len(init_centriod_sets[i]) < K:
            sample = func(L)
            flag = False
            for _, j in enumerate(init_centriod_sets[i]):
                if set(sample) == set(j):
                    flag = True
                    break
            if flag == True:
                continue
            else:
                init_centriod_sets[i].append(sample)

    Largest_sum_list = []
    time_elapsed_list = []
    partition_list = []
    attribute_list = []

    for i in range(init_num):
        time, largest_sum, centriod, partitions = random_sampling(data, init_centriod_sets[i], L, K)
        Largest_sum_list.append(largest_sum)
        time_elapsed_list.append(time)
        partition_list.append(partitions)
        attribute_list.append(centriod)

    return Largest_sum_list, time_elapsed_list, attribute_list, partition_list

if __name__ == "__main__": 
    data = np.random.randint(0, 5, (4, 5))
    print(data)
    data = np.load('../data/netflix/netflix-200.npy')
    Largest_sum_list, time_elapsed_list, attribute_list, partition_list = multiple_random_sampling(data, L = 3, K = 15, init_num = 50)
    print(np.max(Largest_sum_list))
