import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN


def Read_Dataset(path_csv_file, is_unlabel=False):

    df = pd.read_csv(path_csv_file)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    list_features = df.columns

    if is_unlabel == False:
        X = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()
        return (X, y, list_features)

    else:
        return (df.to_numpy(), list_features)

    

def Split_Dataset_Clients(X, y, num_clients):
    
    list_client_X = np.array_split(X, num_clients)
    list_client_y = np.array_split(y, num_clients)

    return (list_client_X, list_client_y) 



def Handle_ImBalance(X, y, sampling_strategy=None):

    print("Before sampling: ", Counter(y))

    if sampling_strategy == None:
        smote_enn = SMOTEENN(sampling_strategy=0.5, random_state=42)
        X, y = smote_enn.fit_resample(X, y)
    else:
        oversampler = RandomOverSampler(sampling_strategy=sampling_strategy)
        X, y = oversampler.fit_resample(X, y)

    print("After sampling: ", Counter(y))
    return (X, y)