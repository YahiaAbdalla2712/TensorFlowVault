import numpy as np

def mae(y_true,y_predicted):
    return np.mean(np.abs(y_predicted - y_true))