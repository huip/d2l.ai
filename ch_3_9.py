from mxnet import nd
from mxnet import gluon
from mxnet.gluon import loss as gloss
import utils


batch_size = 256

train_iter, test_iter = utils.load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens))

b1 = nd.zeros(num_hiddens)

W2 = nd.random.normal(scale=0.01, shape=(num_hiddens, num_outputs))

b2 = nd.zeros(num_outputs)

params = [W1, b1, W2, b2]


# 对参数进行梯度附着
for param in params:
    param.attach_grad()


# 定义激活函数
def relu(X):
    return nd.maximum(X, 0)

# 定义模型
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(nd.dot(X, W1) + b1)
    return nd.dot(H, W2) + b2

loss = gluon.loss.SoftmaxCrossEntropyLoss()

num_epochs, lr = 15, 0.02

utils.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, lr=lr, params=params)



