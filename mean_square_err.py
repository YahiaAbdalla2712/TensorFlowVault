def mse(y_true,y_pred):
    n = len(y_true)
    total = 0
    for i in range(n):
        total += (y_true[i] - y_pred[i])**2
    return total / n    