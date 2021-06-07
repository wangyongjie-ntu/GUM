import numpy as np
import pandas as pd
import random
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import torch
from captum.attr import IntegratedGradients

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

def positive_sample(X, Y):
    
    #X = torch.from_numpy(X).float()
    positive = np.argwhere(Y == 1).squeeze()
    X_positive = X[positive]
    negative = np.argwhere(Y == 0).squeeze()
    X_negative = X[negative]
    return X_positive, X_negative

def train_test(X_train, X_test, Y_train, Y_test):

    # convert numpy array to tensor
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    Y_train = torch.from_numpy(Y_train).long()
    Y_test = torch.from_numpy(Y_test).long()
    model = torch.nn.Sequential(
              torch.nn.Linear(X_train.size(1), 20),
              torch.nn.ReLU(),
              #torch.nn.Dropout(0.3),
              torch.nn.Linear(20, 20),
              torch.nn.ReLU(),
              #torch.nn.Dropout(0.3),
              torch.nn.Linear(20, 10),
              torch.nn.ReLU(),
              #torch.nn.Dropout(0.3),
              torch.nn.Linear(10, 2),
            )
    loss_fn = torch.nn.CrossEntropyLoss()    

    #optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum = 0.2)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, weight_decay = 1e-3)

    for t in range(500):
      # Forward pass: compute predicted y by passing x to the model.
      y_pred = model(X_train)
      # Compute and print loss.
      loss = loss_fn(y_pred, Y_train)
      print(t, loss.item())
      optimizer.zero_grad()
      # Backward pass: compute gradient of the loss with respect to model parameters
      loss.backward()
      # Calling the step function on an Optimizer makes an update to its parameters
      optimizer.step()

    _, y_pred = torch.max(y_pred, 1)
    print("Train accuracy score", accuracy_score(Y_train, y_pred))
    print("Train F1 score", f1_score(Y_train, y_pred))
    output = model(X_test)
    _, y_pred = torch.max(output, 1)
    print("Test accuracy score", accuracy_score(Y_test, y_pred))
    print("Test F1 score", f1_score(Y_test, y_pred))

    return model

def integrated_gradient(model, X, baseline):
    ig = IntegratedGradients(model)
    attr, delta = ig.attribute(X, baseline, target = 1, return_convergence_delta = True)
    attr = attr.detach().numpy()
    return attr

if __name__ == "__main__":

    X_train, X_test, Y_train, Y_test, X, Y, attributes, scaler = titanic1("../data/train.csv")

    if True:
        model = torch.load("titanic_v1.pkl")
    else:
        model = train_test(X_train, X_test, Y_train, Y_test)
        torch.save(model, "titanic_v1.pkl")

    X_positive, X_negative = positive_sample(X, Y)
    X_positive = scaler.transform(X_positive)
    X_negative = scaler.transform(X_negative)
    X_positive = torch.from_numpy(X_positive).float()
    X_negative = torch.from_numpy(X_negative).float()

    idx1 = torch.where(model(X_positive).argmax(1) == 1)
    idx2 = torch.where(model(X_negative).argmax(1) == 0)
    true_positive = X_positive[idx1]
    true_negative = X_negative[idx2]
    baseline = torch.mean(true_negative, 0)
    baseline = baseline.repeat(len(true_positive)).reshape(true_positive.shape)
    importance = integrated_gradient(model, true_positive, baseline)
    np.save("titanic_importance.npy", importance)
    print(importance.mean(0))

