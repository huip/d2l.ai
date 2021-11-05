from mxnet import gluon
from mxnet.gluon.nn.basic_layers import Dense
from mxnet.ndarray.random import shuffle
import matplotlib.pyplot as plt
from mxnet import nd, autograd
from IPython import display
from mxnet.gluon import data as gdata
from mxnet.gluon import nn
from mxnet import init
from mxnet.gluon import loss as gloss

num_inputs =  2
num_examples = 10
true_w = [2, -3.4]
true_b = 4.2

# Generate datas
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)

batch_size = 10
dataset = gdata.ArrayDataset(features, labels)
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)


net = nn.Sequential()


net.add(Dense(1))

net.initialize(init.Normal(sigma=0.01))

loss = gloss.L2Loss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})

num_epochs = 1000

for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))

dence = net[0]

print(true_w, dence.weight.data())
print(true_b, dence.bias.data())


