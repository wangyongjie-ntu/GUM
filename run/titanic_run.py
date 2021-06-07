import os
import sys
sys.path.insert(0, "../")
from algorithm.kmax import *
from algorithm.baseline_random import *
from algorithm.celf import *
from algorithm.baseline_greedy import *
from algorithm.stochastic_lazy import *

#np.random.seed(0)

if __name__ == "__main__":

    if os.path.isfile("../data/titanic_importance_v1.npy"):
        data = np.load("../data/titanic_importance_v1.npy")

    k_list = [1, 2, 3, 4, 5, 6, 7, 8]
    l_list = [2, 3]
    length = len(k_list) * len(l_list)
    data_length = data.shape[0]

    labels = []
    celf_utility = np.zeros(length)
    kmax_utility_max = np.zeros(length)
    kmax_utility_mean = np.zeros(length)
    kmax_utility_std = np.zeros(length)
    baseline_random_utility = np.zeros(length)
    baseline_greedy_utility = np.zeros(length)
    baseline_kma_utility = np.zeros(length)
    sgreedy5_utility_mean = np.zeros(length)
    sgreedy5_utility_std = np.zeros(length)
    sgreedy9_utility_mean = np.zeros(length)
    sgreedy9_utility_std = np.zeros(length)

    for i in range(len(l_list)):
        l = l_list[i]
        for j in range(len(k_list)):
            k = k_list[j]
            # greedy
            time.sleep(0.1)
            _time, largest_sum, SubsetC, partitions = CELF(data, l , k)
            celf_utility[len(k_list) * i + j] = largest_sum / (len(data) * l)

            # kmm and kma
            time.sleep(0.1)
            sum_list, time_list, iteration_list, _, _, increment_list = multiple_init_kmax(data, l, k, init_num = 50)
            kmax_utility_max[len(k_list) * i + j] = max(sum_list)  / (len(data) * l)
            kmax_utility_mean[len(k_list) * i + j] = np.mean(sum_list) / (len(data) * l)
            kmax_utility_std[len(k_list) * i + j] = np.std(sum_list) / (len(data) * l)
            
            # greedy baseline
            time.sleep(0.1)
            time1, largest_sum, itemset, partitions = baseline_greedy(data, l, k)
            baseline_greedy_utility[len(k_list) * i + j] = largest_sum / (len(data) * l)

            # greedy initialization
            time.sleep(0.1)
            _, time2, utility, itemset, partitions, increment = kmax(data, itemset, l, k)
            baseline_kma_utility[len(k_list) * i + j] = utility / (len(data) * l)

            # random selection
            Largest_sum_list, time_elapsed_list, attribute_list, partition_list = multiple_random_sampling(data, l, k, init_num = 50)
            baseline_random_utility[len(k_list) * i + j] = np.mean(Largest_sum_list) / (len(data) * l)

            # stochastic greedy with lazy evaluation
            time.sleep(0.1)
            tmp5_utility = []
            tmp5_time = []
            for _ in range(10):
                _time, utility, itemset, partitions = stochastic_greedy_lazy(data, 0.5, l, k)
                tmp5_utility.append(utility)
                tmp5_time.append(_time)

            sgreedy5_utility_mean[len(k_list) * i + j] = np.mean(tmp5_utility) / (len(data) * l)
            sgreedy5_utility_std[len(k_list) * i + j] = np.std(tmp5_utility) / (len(data) * l)


            # stochastic greedy with lazy evaluation
            time.sleep(0.1)
            tmp9_utility = []
            tmp9_time = []
            for _ in range(10):
                _time, utility, itemset, partitions = stochastic_greedy_lazy(data, 0.9, l, k)
                tmp9_utility.append(utility)
                tmp9_time.append(_time)

            sgreedy9_utility_mean[len(k_list) * i + j] = np.mean(tmp9_utility) / (len(data) * l)
            sgreedy9_utility_std[len(k_list) * i + j] = np.std(tmp9_utility) / (len(data) * l)
            time.sleep(0.1)

            label = "k={}".format(k)
            labels.append(label)
    
    np.save("titanic-kmax-max-utility.npy", kmax_utility_max)
    np.save("titanic-kmax-mean-utility.npy", kmax_utility_mean)
    np.save("titanic-kmax-std-utility.npy", kmax_utility_std)
    np.save("titanic-celf-utility.npy", celf_utility)

    np.save("titanic-smart-init-utility.npy", baseline_greedy_utility)
    np.save("titanic-kms-utility.npy", baseline_kma_utility)
    np.save("titanic-random-init-utility.npy", baseline_random_utility)

    np.save("titanic-sgreedy-0.5-utility.npy", sgreedy5_utility_mean)
    np.save("titanic-sgreedy-0.5-utility-std.npy", sgreedy5_utility_std)
    np.save("titanic-sgreedy-0.9-utility.npy", sgreedy9_utility_mean)
    np.save("titanic-sgreedy-0.9-utility-std.npy", sgreedy9_utility_std)
