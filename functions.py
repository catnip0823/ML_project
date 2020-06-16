def RSS(predict, label):
    error = 0
    for i in range(len(predict)):
        error += (predict[i] - label[i])**2
    return error

