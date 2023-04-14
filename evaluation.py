import os
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from pytorch_tabnet.metrics import Metric

class RocAuc(Metric):
    def __init__(self):
        self._name = "roc_auc"
        self._maximize = True

    def __call__(self, y_true, y_score):
        auc = roc_auc_score(y_true, y_score, multi_class='ovr')
        return auc

class Accuracy(Metric):
    def __init__(self):
        self._name = "accuracy"
        self._maximize = True

    def __call__(self, y_true, y_score):
        accuracy = metrics.accuracy_score(y_true, y_score)
        return accuracy


def Evaluate_Model_Classifier(y_test, y_pred):

    accuracy = metrics.accuracy_score(y_test, y_pred)
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    classification = metrics.classification_report(y_test, y_pred)
    print()
    print('============================== Model Evaluation ==============================')
    print()
    print ("Model Accuracy:" "\n", accuracy)
    print()
    print("Confusion matrix:" "\n", confusion_matrix)
    print()
    print("Classification report:" "\n", classification) 
    print()


def Calculate_Loss_and_Accuracy(y_test, y_pred_proba):
    accuracy = metrics.accuracy_score(y_test, np.argmax(y_pred_proba, axis=1))
    loss = metrics.log_loss(y_test, y_pred_proba) 
    f_1 = metrics.f1_score(y_test, np.argmax(y_pred_proba, axis=1), average='macro')

    return (accuracy, loss, f_1)