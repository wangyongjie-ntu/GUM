import numpy as np 
import random
import time
import copy

def kmax(data, init_centriod, L = 2, K =  3, max_iterations = 50):
    '''
    data: a collection of records
    init_centriod: the inital centriod
    L: top K attributes
    K: cluster numbers
    max_iterations: the max iteration before stop
    '''
    start = time.time()
    length, width = data.shape
    partitions = [[] for _ in range(K)]
    attribute_sum = np.zeros((K, width))
    largest_sum = 0
    increment = []
    for iteration in range(max_iterations):
        # data partition
        tmp_p = copy.deepcopy(partitions)
        _sum = 0
        tmp_sum = np.zeros((length, len(init_centriod)))
        selected_data = np.take(data, init_centriod, axis = 1)
        for k in range(K):
            #selected_attr = data[:, init_centriod[i]]
            #tmp_sum[:, i] = np.sum(selected_attr, axis = 1)
            tmp_sum[:, k] = selected_data[:, k, :].sum(1)

        tmp = np.argmax(tmp_sum, 1)
        increment.append(np.sum(np.max(tmp_sum, 1)))# / (length * K))
        for k in range(K):
            partitions[k] = np.where(tmp == k)[0].tolist()
            attribute_sum[k] = np.sum(data[partitions[k], :], 0)

        # recompute the centriod
        for k in range(K):
            _sum += sum(np.sort(attribute_sum[k], kind = "stable")[-L:])
            tmp_idx = np.argsort(attribute_sum[k], kind = "stable")
            new_centriod = tuple(tmp_idx[-L:].tolist())
            init_centriod[k] = new_centriod
            
        largest_sum = _sum
        #increment.append(iteration_sum / (length * K))
        Flag = True
        for k in range(K):
            Flag = Flag & (tmp_p[k] == partitions[k])

        if Flag == True:
            break

    end = time.time()
    return iteration + 1, end - start, largest_sum, init_centriod, partitions, increment
    
def multiple_init_kmax(data, L = 2, K = 3, init_num = 1):
    '''
    data: a collection of records
    init_centriod: the inital centriod
    L: top L attributes
    K: cluster numbers
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
    iteration_list = []
    partition_list = []
    attribute_list = []
    increment_list = []

    for i in range(init_num):
        iteration, time, largest_sum, centriod, partitions, increment = kmax(data, init_centriod_sets[i], L, K)
        Largest_sum_list.append(largest_sum)
        time_elapsed_list.append(time)
        iteration_list.append(iteration)
        partition_list.append(partitions)
        attribute_list.append(centriod)
        increment_list.append(increment)

    return Largest_sum_list, time_elapsed_list, iteration_list, attribute_list, partition_list, increment_list

if __name__ == "__main__":
    data = np.load("../titanic/titanic_importance_neg.npy")
    print(data)
    for i in range(1, 10):
        largest_sum, time_elapsed, _, _, _, _ = multiple_init_kmax(data, 2, i, init_num = 10)
        print("Largest sum\t", np.max(largest_sum))
