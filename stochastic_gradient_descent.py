import numpy as np
import random

def stochastic_grad_descent(X, y_true, epochs, learning_rate=0.01):

    number_of_features = X.shape[1]
    w = np.ones(shape=number_of_features)
    b = 0

    total_samples = X.shape[0]

    cost_list = []
    epoch_list = []

    for i in range(epochs):

        random_index = random.randint(0, total_samples - 1)

        sample_x = X[random_index]
        sample_y = y_true[random_index]

        y_predicted = np.dot(w, sample_x) + b

        error = sample_y - y_predicted

        w_grad = -(2) * sample_x * error
        b_grad = -(2) * error

        w = w - learning_rate * w_grad
        b = b - learning_rate * b_grad

        cost = np.mean(np.square(error))

        if i % 10 == 0:
            cost_list.append(cost)
            epoch_list.append(i)

    return w, b, cost, cost_list, epoch_list