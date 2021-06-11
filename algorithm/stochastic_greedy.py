import numpy as np
import itertools
import time
import math
import random

def stochastic_greedy_lazy(data, epsilon = 0.5, L = 2, K = 3):
    '''
    data: a collection of records
    K: top K attributes
    C: cluster numbers
    '''

    start = time.time()
    length, width = data.shape
    attributes_sets = list(itertools.combinations(range(width), L))
    sample_num =  int(np.ceil(len(attributes_sets) / K * math.log(1 / epsilon)))

    SubsetC = []
    partitions = [[] for _ in range(K)]
    largest_sum = 0

    delta_list = [2**31 - 1] * len(attributes_sets)

    for i in range(0, K):
        all_index = range(len(attributes_sets))
        sample_num = min(sample_num, len(attributes_sets))
        sample_index = random.sample(all_index, sample_num)
        marginal_gain = [delta_list[n] for n in sample_index]
        sample_attributes_set = [attributes_sets[m] for m in sample_index]
        sum_list = np.zeros(len(sample_attributes_set))
        tmp_sum = np.zeros((length, len(SubsetC) + 1))
        for m in range(len(SubsetC)):
            selected_attr = data[:, SubsetC[m]]
            tmp_sum[:, m] = np.sum(selected_attr, axis = 1)
        
        max_marginal_gain = -10000
        for j in range(len(sample_index)):
            if max_marginal_gain >= marginal_gain[j]:
                continue
            else:
                current_set = sample_attributes_set[j]
                tmp_sum[:, -1] = np.sum(data[:, current_set], axis = 1)
                maximum = np.max(tmp_sum, axis = 1)
                sum_list[j] = np.sum(maximum)
                marginal_gain[j] = sum_list[j] - largest_sum
                delta_list[sample_index[j]] = sum_list[j] - largest_sum
                if sum_list[j] - largest_sum > max_marginal_gain:
                    max_marginal_gain = sum_list[j] - largest_sum

        idx = np.argmax(sum_list)
        largest_sum = sum_list[idx]
        SubsetC.append(sample_attributes_set[idx])
        attributes_sets.remove(sample_attributes_set[idx])
        delta_list.pop(sample_index[idx])
        #delta_list.remove(marginal_gain[idx])

    #obtain the partitions under subsetC
    tmp_sum = np.zeros((length, len(SubsetC)))
    for m in range(len(SubsetC)):
        selected_attr = data[:, SubsetC[m]]
        tmp_sum[:, m] = np.sum(selected_attr, axis = 1)

    tmp = np.argmax(tmp_sum, 1)
    for i in range(K):
        partitions[i] = np.where(tmp == i)[0].tolist()
          
    end = time.time()
    return end - start, largest_sum, SubsetC, partitions

if __name__ == "__main__":
    
    data = np.random.randint(0, 5, (60000, 200))
    print(data)
    time_elapsed, largest_sum, SubsetC, partitions = stochastic_greedy_lazy(data, L = 2, K = 50)
    print(largest_sum)
    print(time_elapsed)
    print(SubsetC)

