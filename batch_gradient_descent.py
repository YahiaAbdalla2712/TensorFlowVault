import numpy as np
from sklearn import preprocessing


def batch_grad_descent(features, y_true, epochs, learning_rate = 0.01):
    number_of_features = features.shape[1]
    w = np.ones(shape = number_of_features)
    b = 0
    total_samples = features.shape[0]
    s = preprocessing.MinMaxScaler()
    scaled_features = s.fit_transform(features)

    cost_list = []
    epoch_list = []

    for i in range(epochs):
        y_predicted = np.dot(w,scaled_features.T) + b 
        w_grad = -(2/total_samples)*(scaled_features.T.dot(y_true - y_predicted))
        b_grad = -(2/total_samples)*np.sum(y_true-y_predicted)
        w = w - learning_rate * w_grad
        b = b - learning_rate * b_grad
        cost = np.mean(np.square(y_true-y_predicted))
        if i%10 == 0:
            cost_list.append(cost)
            epoch_list.append(i)
    return w, b, cost, cost_list, epoch_list                

