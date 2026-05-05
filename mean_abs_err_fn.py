def mae(y_true, y_predicted):
    total_err = 0
    for yt, yp in zip(y_true, y_predicted):
        total_err += abs(yt - yp)
    print("Total error:",total_err)
    mae = total_err/len(y_true)
    print("MAE:",mae)
    return mae    