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

    if os.path.isfile("../data/netflix-200.npy"):
        data = np.load("../data/netflix-200.npy")

    k_list = [5, 10, 15, 20, 25, 30]
    l_list = [2, 3]
    length = len(k_list) * len(l_list)
    data_length = data.shape[0]

    celf_utility = np.zeros(length)
    celf_time = np.zeros(length)
    kmax_utility_max = np.zeros(length)
    kmax_time_max = np.zeros(length)
    kmax_utility_mean = np.zeros(length)
    kmax_time_mean = np.zeros(length)
    kmax_utility_std = np.zeros(length)
    kmax_time_std = np.zeros(length)
    baseline_random_utility = np.zeros(length)
    baseline_random_time = np.zeros(length)
    baseline_greedy_utility = np.zeros(length)
    baseline_greedy_time = np.zeros(length)
    baseline_kms_utility = np.zeros(length)
    baseline_kms_time = np.zeros(length)
    sgreedy5_utility_mean = np.zeros(length)
    sgreedy5_time = np.zeros(length)
    sgreedy9_utility_mean = np.zeros(length)
    sgreedy9_time = np.zeros(length)

    for i in range(len(l_list)):
        l = l_list[i]
        for j in range(len(k_list)):
            k = k_list[j]
            print("L= {}, K = {}".format(l, k))
            # greedy
            time.sleep(0.1)
            if l < 3:
                _time, largest_sum, SubsetC, partitions = CELF(data, l , k)
                celf_utility[len(k_list) * i + j] = largest_sum / (len(data) * l)
                celf_time[len(k_list) * i + j] = _time

            # kmm and kma
            time.sleep(0.1)
            sum_list, time_list, iteration_list, _, _, increment_list = multiple_init_kmax(data, l, k, init_num = 50)
            kmax_utility_max[len(k_list) * i + j] = max(sum_list)  / (len(data) * l)
            kmax_utility_mean[len(k_list) * i + j] = np.mean(sum_list) / (len(data) * l)
            kmax_utility_std[len(k_list) * i + j] = np.std(sum_list) / (len(data) * l)
            kmax_time_max[len(k_list) * i + j] = np.sum(time_list)
            kmax_time_mean[len(k_list) * i + j] = np.mean(time_list)
            kmax_time_std[len(k_list) * i + j] = np.std(time_list)
            
            # greedy baseline
            time.sleep(0.1)
            time1, largest_sum, itemset, partitions = baseline_greedy(data, l, k)
            baseline_greedy_utility[len(k_list) * i + j] = largest_sum / (len(data) * l)
            baseline_greedy_time[len(k_list) * i + j] = time1

            # greedy initialization
            time.sleep(0.1)
            _, time2, utility, itemset, partitions, increment = kmax(data, itemset, l, k)
            baseline_kms_utility[len(k_list) * i + j] = utility / (len(data) * l)
            baseline_kms_time[len(k_list) * i + j] = time1 + time2

            # random selection
            Largest_sum_list, time_elapsed_list, attribute_list, partition_list = multiple_random_sampling(data, l, k, init_num = 50)
            baseline_random_utility[len(k_list) * i + j] = np.mean(Largest_sum_list) / (len(data) * l)
            baseline_random_time[len(k_list) * i + j] = np.mean(time_elapsed_list)

            # stochastic greedy with lazy evaluation
            if l < 3:
                _time, utility, itemset, partitions = stochastic_greedy_lazy(data, 0.5, l, k)
                sgreedy5_utility_mean[len(k_list) * i + j] = utility / (len(data) * l)
                sgreedy5_time[len(k_list) * i + j] = _time


            # stochastic greedy with lazy evaluation
            _time, utility, itemset, partitions = stochastic_greedy_lazy(data, 0.9, l, k)
            sgreedy9_utility_mean[len(k_list) * i + j] = utility / (len(data) * l)
            sgreedy9_time[len(k_list) * i + j] = _time

    
    np.save("netflix-200-kmax-max-utility.npy", kmax_utility_max)
    np.save("netflix-200-kmax-mean-utility.npy", kmax_utility_mean)
    np.save("netflix-200-kmax-std-utility.npy", kmax_utility_std)
    np.save("netflix-200-celf-utility.npy", celf_utility)

    np.save("netflix-200-smart-init-utility.npy", baseline_greedy_utility)
    np.save("netflix-200-kms-utility.npy", baseline_kma_utility)
    np.save("netflix-200-random-init-utility.npy", baseline_random_utility)

    np.save("netflix-200-sgreedy-0.5-utility.npy", sgreedy5_utility_mean)
    np.save("netflix-200-sgreedy-0.5-utility-std.npy", sgreedy5_utility_std)
    np.save("netflix-200-sgreedy-0.9-utility.npy", sgreedy9_utility_mean)
    np.save("netflix-200-sgreedy-0.9-utility-std.npy", sgreedy9_utility_std)
