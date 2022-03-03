import mxnet as mx
from mxnet import nd, gluon, autograd, init
from mxnet.gluon import loss as gloss, nn, trainer
import time
import utils


net = nn.Sequential()

net.add(
    nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
    nn.MaxPool2D(pool_size=3, strides=2),
    nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
    nn.MaxPool2D(pool_size=3, strides=2),
    # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
        # 前两个卷积层后不使用池化层来减小输入的高和宽
    nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
    nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
    nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'),
    nn.MaxPool2D(pool_size=3, strides=2),
    # 这里全连接层的输出个数比LeNet中的大数倍。使用丢弃层来缓解过拟合
    nn.Dense(4096, activation="relu"), nn.Dropout(0.5),
    nn.Dense(4096, activation="relu"), nn.Dropout(0.5),
    # 输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
    nn.Dense(10)
)

X = nd.random.uniform(shape=(1, 1, 224, 224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)


batch_size = 64

print(111)

train_iter, test_iter = utils.load_data_fashion_mnist(batch_size=batch_size, resize=224)

print(112)
lr, num_epochs = 0.99, 10

net.initialize(force_reinit=True, init=init.Xavier())

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
utils.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, trainer)