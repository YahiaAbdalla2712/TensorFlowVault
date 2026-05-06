import numpy as np

def log_loss(y_true, y_predicted):
    epsilon = 1e-15
    y_predicted_new = np.clip(y_predicted, epsilon, 1-epsilon)
    return -np.mean(y_true*np.log(y_predicted_new)+(1-y_true)* np.log(1-y_predicted_new))