from mxnet import autograd, nd
from mxnet.gluon import nn
from numpy.core.fromnumeric import shape

def corr2d(X, K):
    h, w = K.shape
    Y = nd.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    
    return Y

X = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = nd.array([[0, 1], [2, 3]])

Y = corr2d(X, K)



# 图像边缘检测
X = nd.ones((6, 8))
X[:, 2:6] = 0


K = nd.array([[1, -1]])

Y = corr2d(X, K)


class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))
    
    def forward(self, x):
        return corr2d(x, self.weight.data) + self.bias


conv2d =  nn.Conv2D(1, kernel_size=(1, 2))

conv2d.initialize()

# 二维卷积层使用4维输入输出，格式为(样本, 通道, 高, 宽)，这里批量大小（批量中的样本数）和通
# 道数均为1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))

for i in range(10):
    with autograd.record():
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
    l.backward()

    conv2d.weight.data()[:] -= 3e-2 * conv2d.weight.grad()
    if (i + 1) % 2 == 0:
        print('batch %d, loss %.3f' % (i + 1, l.sum().asscalar()))

print(conv2d.weight.data().reshape((1, 2)))


def comp_conv2d(conv2d, X):
    conv2d.initialize()
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    print(Y.shape)
    return Y.reshape(Y.shape[2:])

conv2d = nn.Conv2D(1, kernel_size=3, padding=1)

X = nd.random.uniform(shape=(8, 8))

result = comp_conv2d(conv2d, X)

print(result.shape)