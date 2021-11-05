from mxnet import gluon, nd, init
from mxnet.gluon import loss as gloss, nn
import utils

batch_size = 256

train_iter, test_iter = utils.load_data_fashion_mnist(batch_size)


# 模型初始化
net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))

# 定义损失函数
loss = gloss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})

utils.train_ch3(net, train_iter, test_iter, loss, num_epochs=100, batch_size=batch_size,trainer=trainer)

