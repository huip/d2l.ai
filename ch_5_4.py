from mxnet import nd
from mxnet.gluon import nn

def pool2d(X, pool_size, mode='max'):
    h, w = pool_size
    Y = nd.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = (X[i: i + h, j: j + w]).max()
            elif mode == 'avg':
                Y[i, j] = (X[i: i + h, j: j + w]).mean()
    return Y


X = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

print(pool2d(X, (2, 2), mode='avg'))