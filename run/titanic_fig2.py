import os
import pandas as pd
import sys
sys.path.insert(0, '..')
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from algorithm.kmax import *
import matplotlib.pyplot as plt

def scatter_plot(fea_importance, attributes, SubsetC, best_partitions, title, name):
    '''
    fea_importance: the feature importance obtained by integrated gradient
    X: the inputs features
    attributes: the feature names
    SubsetC: the Subset achieved by greedy or kmeans algorithm
    best_partitions: the data partitions
    title: the title of figure
    name: figure name
    '''
    importance_mean = np.mean(fea_importance, 0)
    X = np.arange(len(importance_mean)) 
    plt.clf()
    fig, ax = plt.subplots()
    #ax.scatter(X, importance_mean, marker = "o", c = "orange", linewidths = 8, label = "total:" + str(len(fea_importance)))
    ax.scatter(X, importance_mean, marker = "o", c = "orange", linewidths = 8)

    markers = ["^", "s", "P", "*", "X", "8"]
    colors = ['blue', 'green', 'red', 'purple', 'olive', 'cyan']

    for i in range(len(SubsetC)):
        indice = SubsetC[i]
        cluster = fea_importance[best_partitions[i]]
        if type(best_partitions[i]) != int:
            cluster_mean = np.mean(cluster, 0)
        else:
            cluster_mean = cluster

        tmp = cluster_mean[[indice]]
        if type(best_partitions[i]) != int:
            label = "X" + str(i + 1) + ":" + str(len(best_partitions[i]))
        else:
            label = "X" + str(i) + ":" + str(1)
        
        print(tmp, indice)
        print(len(best_partitions[i]))
        ax.scatter(indice, tmp, marker = markers[i], c = colors[i], linewidths = 8, label = label)

    for i in X:
        ax.axvline(i, color = "lightsteelblue")

    plt.legend(ncol = 5, loc = "lower center", bbox_to_anchor = (0.5, 1.02), fontsize = 15)
    plt.ylabel("Group item utility", fontsize = 16)
    plt.xticks(X, attributes, fontsize = 15)
    plt.tight_layout()
    plt.savefig(name)

def titanic1(filename = "./data/titanic/train.csv"):

    df = pd.read_csv(filename)
    # drop name and ID
    df = df.drop(['Name', 'PassengerId'], axis=1)
    # Sex
    df['Sex'] = df['Sex'].map({'female': 1, 'male': 0} ).astype(int)
    # the median value 
    median_age = np.zeros((2,3))
    for sex in range(0, 2):
        for pclass in range(0, 3):
            guess_df = df[(df['Sex'] == sex) & (df['Pclass'] == pclass+1)]['Age'].dropna()
            age_guess = guess_df.median()
            median_age[sex, pclass] = age_guess

    for i in range(0, 2):
        for j in range(0, 3):
            df.loc[(df.Age.isnull()) & (df.Sex == i) & (df.Pclass == j+1), 'Age'] = median_age[i,j]
    
    #df['Age'] = df['Age'].astype(int)
    df = df.drop(['Ticket', 'Cabin'], axis=1)
    freq_port = df.Embarked.dropna().mode()[0]
    df['Embarked'] = df['Embarked'].fillna(freq_port)
    df = pd.get_dummies(df, columns = ['Embarked'])
    df.rename(columns = {"Embarked_C":"C",
        "Embarked_Q":"Q",
        "Embarked_S":"S"}, inplace = True)
    #df['Embarked'] = df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    Y = df['Survived'].values
    X = df.drop("Survived", axis = 1)
    columns = X.columns
    X = X.values
    df = df.to_numpy()
    scaler = StandardScaler()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, Y_train, Y_test, X, Y, columns, scaler

if __name__ == "__main__":

    X_train, X_test, Y_train, Y_test, X, Y, attributes, scaler = titanic1("../data/train.csv")
    K, C = 2, 3
    dataset = "titanic" + "_" + str(K) + "_" + str(C)
    title = dataset
    saved_name = dataset + ".png"
    if os.path.isfile("../data/titanic_importance_v1.npy"):
        importance = np.load("../data/titanic_importance_v1.npy")

    sum_list, time_list, iteration_list, attribute_list, partition_list, increment_list  = multiple_init_kmax(importance, K, C, init_num = 50)
    #print(increment_list)
    length = [len(increment) for increment in increment_list]
    max_length = np.argmax(length)
    increment = increment_list[max_length]
    max_idx = np.argsort(sum_list)[-1]
    print(np.max(sum_list))
    SubsetC = attribute_list[max_idx]
    partition = partition_list[max_idx]
    np.save(dataset + "_subset.npy", SubsetC)
    np.save(dataset + "_partition.npy", partition)

    scatter_plot(importance, attributes, SubsetC, partition, title = dataset, name = saved_name)
