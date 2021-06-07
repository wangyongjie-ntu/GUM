import numpy as np
import time
import random
import itertools

def baseline_greedy(data, L = 2, K = 3):

    # data: a collection of records
    # K: top K attributes
    # C: cluster numbers
    start = time.time()
    length, width = data.shape
    T = np.copy(data)
    partitions = [[] for _ in range(K)]
    largest_sum = 0
    itemset = []
    for i in range(K):
        if T.shape[0] == 0:
            break
        
        #print(T.shape)
        column_sum = np.sum(T, axis = 0)
        sorted_index = np.argsort(column_sum, kind = 'stable')[::-1]
        attribute_sets = itertools.combinations(sorted_index, L)
        for _, index in enumerate(attribute_sets):
            flag = False
            for _, j in enumerate(itemset):
                if set(index) == set(j):
                    flag = True
                    break
            if flag == True:
                continue
            else:
                break

        itemset.append(index)
        #while frozenset(index) in 
        tmp = np.argsort(T, axis = 1, kind = 'stable')[:, -L:]
        #tmp = np.argpartition(T, axis = 1, kind = 'introselect', kth = -L)[:, -L:]
        topL_data = np.take_along_axis(T, tmp, axis = 1)
        topL_sum = topL_data.sum(1)
        utility_index = np.take(T, index, axis = 1).sum(1)
        non_empty = np.argwhere((topL_sum == utility_index) == 0)
        non_empty  = np.squeeze(non_empty)
        T = T[non_empty]

    tmp_sum = np.zeros((length, len(itemset)))
    selected_data = np.take(data, itemset, axis = 1)
    for k in range(len(itemset)):
        tmp_sum[:, k] = selected_data[:, k, :].sum(1)
    
    tmp = np.argmax(tmp_sum, 1)
    largest_sum = np.sum(np.max(tmp_sum, 1))
    for k in range(len(itemset)):
        partitions[k] = np.where(tmp == k)[0].tolist()

    end = time.time()
    end = time.time()
    return end - start, largest_sum, itemset, partitions

if __name__ == "__main__":
    #data = np.load("../titanic/titanic_importance_neg.npy")
    data = np.load("../data/netflix/netflix-200.npy")
    time_elapsed, largest_sum, itemset, partitions = baseline_greedy(data, L = 5, K = 30)
    '''
    for i in range(3, 9):
        time_elapsed, largest_sum, itemset, partitions = baseline_greedy(data, L = 3, K = i)
        print("Time elapsed\t", time_elapsed)
        print("Largest sum\t", largest_sum)
        print("SubSetC\t", itemset)
    '''
