'''
metric functions 
Args:
    pred, label: numpy array of prediction, label
'''
import sklearn.metrics
import numpy as np


def r2_score(pred, label):
    return sklearn.metrics.r2_score(label, pred)


def multiclass_accuracy(pred, label):
    pred_i = np.argmax(pred, axis=1)
    return np.sum(pred_i == label)/label.shape[0]
