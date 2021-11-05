from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn
import utils

net = nn.Sequential()

net.add(nn.Dense(256, activation='relu'), nn.Dense(10))

net.initialize(init.Normal(sigma=0.01))

batch_size = 256

loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size)
num_epochs = 5

utils.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, trainer=trainer)
