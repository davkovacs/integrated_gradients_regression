"""Definition of the neural network models and data loaders """

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston

class Net_shallow(nn.Module):
   # shallow neural network
    def __init__(self, n_feature, width_hidden):
        # n_feature: number input dimensions
        # width_hidden: width of the hidden layers
        super(Net_shallow,self).__init__()
        self.ff1 = nn.Linear(n_feature, width_hidden)  # 1st hidden layer
        self.ff2 = nn.Linear(width_hidden, width_hidden)  # 2nd hidden layer
        self.predict = nn.Linear(width_hidden, 1)  # linear output for regression

    def forward(self, x):
        # backward function automatically defined by autograd
        x = F.relu(self.ff1(x))  # activation for 1st hidden layer
        x = F.relu(self.ff2(x))  # activation for 2nd hidden layer
        x = self.predict(x)  #linear output
        return x

class Net_deep(nn.Module):
   # deep neural network
    def __init__(self, n_feature, width_hidden):
        # n_feature: number of input simensions
        # width_hidden: width of the hidden layers
        super(Net_deep,self).__init__()
        # 10 hidden layers of the same size
        self.ff1 = nn.Linear(n_feature, width_hidden)
        self.ff2 = nn.Linear(width_hidden, width_hidden)
        self.ff3 = nn.Linear(width_hidden, width_hidden)
        self.ff4 = nn.Linear(width_hidden, width_hidden)
        self.ff5 = nn.Linear(width_hidden, width_hidden)
        self.ff6 = nn.Linear(width_hidden, width_hidden)
        self.ff7 = nn.Linear(width_hidden, width_hidden)
        self.ff8 = nn.Linear(width_hidden, width_hidden)
        self.ff9 = nn.Linear(width_hidden, width_hidden)
        self.predict = nn.Linear(width_hidden, 1)

    def forward(self, x):
        # backward function automatically defined by autograd
        x = F.relu(self.ff1(x))  # activation for 1st hidden layer
        x = F.relu(self.ff2(x))  # activation for 2nd hidden layer
        x = F.relu(self.ff3(x))
        x = F.relu(self.ff4(x))
        x = F.relu(self.ff5(x))
        x = F.relu(self.ff6(x))
        x = F.relu(self.ff7(x))
        x = F.relu(self.ff8(x))
        x = F.relu(self.ff9(x))
        x = self.predict(x)  #linear output
        return x

def load_bost_preprocessed():
    # Loading and formatting the data
    boston = load_boston()
    df = pd.DataFrame(boston.data)
    df.columns = boston.feature_names
    df['Price'] = boston.target
    # standardize the data
    data = df[df.columns[:-1]]
    data = data.apply(lambda x: (x - x.mean()) / x.std())
    data['Price'] = df.Price
    return data

def load_concrete_preprocessed():
    # Loading and formatting the data
    concrete = pd.read_csv("/home/cdt1906/Documents/cdt/mphil/MiniProject_2/concrete/yeh-concret-data/Concrete_Data_Yeh.csv")
    df = pd.DataFrame(concrete)
    # standardize the data
    data = df[df.columns[:-1]]
    data = data.apply(lambda x: (x - x.mean()) / x.std())
    data['csMPa'] = df.csMPa
    return data