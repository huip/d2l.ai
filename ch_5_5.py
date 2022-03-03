import mxnet as mx
from mxnet import nd, gluon, autograd, init
from mxnet.gluon import loss as gloss, nn, trainer
import time
import utils

net = nn.Sequential()

net.add(
    nn.Conv2D(channels=6, kernel_size=2, activation='sigmoid'),
    nn.MaxPool2D(pool_size=1, strides=2),
    nn.Conv2D(channels=16, kernel_size=2, activation='sigmoid'),
    nn.MaxPool2D(pool_size=1, strides=2),
    # Dense会默认将(批量大小, 通道, 高, 宽)形状的输入转换成
    # (批量大小, 通道 * 高 * 宽)形状的输入
    nn.Dense(120, activation='sigmoid'),
    nn.Dense(84, activation='sigmoid'),
    nn.Dense(10)
)

X = nd.random.uniform(shape=(1, 1, 28, 28))


batch_size = 256
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size=batch_size)


lr, num_epochs = 0.99, 10

net.initialize(force_reinit=True, init=init.Xavier())

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
utils.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, trainer)
