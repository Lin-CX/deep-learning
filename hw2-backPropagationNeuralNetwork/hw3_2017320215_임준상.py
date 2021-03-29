import sys
import os
import matplotlib.pyplot as plt
import numpy as np


def lenc(temp):                                 # 形状函数
    print(len(temp), len(temp[0]))

class nn_linear_layer:

    # linear layer.
    # randomly initialized by creating matrix W and bias b
    def __init__(self, input_size, output_size, std=1):
        self.W = np.random.normal(0, std, (output_size, input_size))
        self.b = np.random.normal(0, std, (output_size, 1))

    ######
    ## Q1
    def forward(self, x):
        x = np.array(x)
        if len(x) != len(self.W[0]):               # 처음에 들어올 때 x의 shape를 조절해야함
            x = np.reshape(x.T, (-1, len(x)))
        s = self.W @ x + self.b
        return s

    ######
    ## Q2
    ## returns three parameters
    def backprop(self, x, dLdy):
        #lenc(x)        4 20
        #lenc(dLdy)     2 20
        #sizeb = np.shape(self.b)[0]
        gradient = (np.sum(dLdy, axis=1, keepdims=True))
        dLdb = gradient.T
        if len(x) == len(dLdy[0]):
            dLdW = dLdy @ x
        else:
            dLdW = dLdy @ x.T
        dLdx = self.W.T @ dLdy
        #lenc(dLdb)
        #lenc(dLdW)
        #lenc(dLdx)

        #dLdW = np.sum(x, axis=0, keepdims=True) / 20   W是矩阵
        return dLdW, dLdb, dLdx

    def update_weights(self, dLdW, dLdb):
        # parameter update
        self.W = self.W + dLdW
        self.b = self.b + dLdb


class nn_activation_layer:

    def __init__(self):
        pass

    ######
    ## Q3
    def forward(self, x):
        sigm = 1 / (1+np.exp(-x))        # sigmoid
        return sigm

    ######
    ## Q4
    def backprop(self, x, dLdy):
        """sigm = 1 / (1+np.exp(-dLdy))
        x += sigm"""
        #lenc(x)        4 20
        #lenc(dLdy)     4 20
        sigm = 1 / (1+np.exp(-x))
        sigm_deri = sigm * (1-sigm)
        ret = dLdy * (sigm_deri)
        return ret


class nn_softmax_layer:
    def __init__(self):
        pass

    ######
    ## Q5
    def forward(self, x):
        s_exp = np.exp(x - np.max(x))
        #s_exp = np.exp(x)
        s_sum = np.sum(s_exp, axis=0, keepdims=True)
        sc = s_exp / s_sum
        return sc

    ######
    ## Q6
    def backprop(self, x, dLdy):
        #lenc(dLdy)     2 20    gradient
        #lenc(x)        2 20    score

        return dLdy


class nn_cross_entropy_layer:
    def __init__(self):
        pass

    ######
    ## Q7
    def forward(self, x, y):
        CEloss = 0
        n = len(x[0])
        #CEloss = np.sum(-np.log10(x.T[range(n), y.T[0]]))
        for index in range(n):
            CEloss += -np.log10(x[y[index][0]][index])
        return CEloss

    ######
    ## Q8
    def backprop(self, x, y):
        # lenc(x): 2 20
        # lenc(y): 20 1
        # softmax의 loss함수가 cross entropy인 경우 그의 편미분은 pi - yi
        # 여기는 softmax와 cross entropy loss의 편미분을 함계 계산했다.
        # 그래서 nn_softmax_layer함수의 backprop method는 그냥 gradient를 return한다.
        gradient = x
        ylength = len(y)
        yl = y.T.tolist()[0]
        gradient[yl, range(ylength)] -= 1
        return gradient / ylength


# number of data points for each of (0,0), (0,1), (1,0) and (1,1)
num_d = 5

# number of test runs
num_test = 40

## Q9. Hyperparameter setting
## learning rate (lr)and number of gradient descent steps (num_gd_step)
## This part is not graded (there is no definitive answer).
## You can set this hyperparameters through experiments.
lr = 1
num_gd_step = 1500

# dataset size
batch_size = 4 * num_d

# number of classes is 2
num_class = 2

# variable to measure accuracy
accuracy = 0

# set this True if want to plot training data
show_train_data = True

# set this True if want to plot loss over gradient descent iteration
show_loss = True

for j in range(num_test):

    # create training data
    m_d1 = (0, 0)
    m_d2 = (1, 1)
    m_d3 = (0, 1)
    m_d4 = (1, 0)

    sig = 0.05
    s_d1 = sig ** 2 * np.eye(2)

    d1 = np.random.multivariate_normal(m_d1, s_d1, num_d)   # class里面的五个点
    d2 = np.random.multivariate_normal(m_d2, s_d1, num_d)
    d3 = np.random.multivariate_normal(m_d3, s_d1, num_d)
    d4 = np.random.multivariate_normal(m_d4, s_d1, num_d)

    # training data, and has dimension (4*num_d,2,1)
    x_train_d = np.vstack((d1, d2, d3, d4))                                                                     # 将数组堆叠起来 变成20-by-2的数组 由四个5-by-2的数组堆叠而成
    # training data lables, and has dimension (4*num_d,1)
    y_train_d = np.vstack((np.zeros((2 * num_d, 1), dtype='uint8'), np.ones((2 * num_d, 1), dtype='uint8')))    # 20-by-1的数组 表示label 前面十个是0 后面十个是1

    # plotting training data if needed
    if (show_train_data) & (j == 0):
        plt.grid()
        plt.scatter(x_train_d[range(2 * num_d), 0], x_train_d[range(2 * num_d), 1], color='b', marker='o')
        plt.scatter(x_train_d[range(2 * num_d, 4 * num_d), 0], x_train_d[range(2 * num_d, 4 * num_d), 1], color='r',
                    marker='x')
        plt.show()

    # 到这里为止上面是堆叠数组即input + 设定labels 下面开始操作
    # create layers

    # hidden layer
    # linear layer
    layer1 = nn_linear_layer(input_size=2, output_size=4, )
    # activation layer
    act = nn_activation_layer()

    # output layer
    # linear
    layer2 = nn_linear_layer(input_size=4, output_size=2, )
    # softmax
    smax = nn_softmax_layer()
    # cross entropy
    cent = nn_cross_entropy_layer()

    # variable for plotting loss
    loss_out = np.zeros((num_gd_step))              # 给定num_gd_step个0 ([0, 0, 0, ..., 0])

    for i in range(num_gd_step):

        # fetch data
        x_train = x_train_d
        y_train = y_train_d

        # create one-hot vectors from the ground truth labels
        y_onehot = np.zeros((batch_size, num_class))
        y_onehot[range(batch_size), y_train.reshape(batch_size, )] = 1          # label为i时 y_onehot[i] = 1

        ################
        # forward pass

        # hidden layer
        # linear
        l1_out = layer1.forward(x_train)
        # activation
        a1_out = act.forward(l1_out)

        # output layer
        # linear
        l2_out = layer2.forward(a1_out)
        # softmax
        smax_out = smax.forward(l2_out)
        # cross entropy loss
        loss_out[i] = cent.forward(smax_out, y_train)

        ################
        # perform backprop
        # output layer

        # cross entropy
        b_cent_out = cent.backprop(smax_out, y_train)
        # softmax
        b_nce_smax_out = smax.backprop(l2_out, b_cent_out)

        # linear
        b_dLdW_2, b_dLdb_2, b_dLdx_2 = layer2.backprop(x=a1_out, dLdy=b_nce_smax_out)

        # backprop, hidden layer
        # activation
        b_act_out = act.backprop(x=l1_out, dLdy=b_dLdx_2)
        # linear
        b_dLdW_1, b_dLdb_1, b_dLdx_1 = layer1.backprop(x=x_train, dLdy=b_act_out)

        ################
        # update weights: perform gradient descent
        layer2.update_weights(dLdW=-b_dLdW_2 * lr, dLdb=-b_dLdb_2.T * lr)
        layer1.update_weights(dLdW=-b_dLdW_1 * lr, dLdb=-b_dLdb_1.T * lr)

        if (i + 1) % 2000 == 0:
            print('gradient descent iteration:', i + 1)

    # set show_loss to True to plot the loss over gradient descent iterations
    if (show_loss) & (j == 0):
        plt.figure(1)
        plt.grid()
        plt.plot(range(num_gd_step), loss_out)
        plt.xlabel('number of gradient descent steps')
        plt.ylabel('cross entropy loss')
        plt.show()

    ################
    # training done
    # now testing

    predicted = np.ones((4,))

    # predicting label for (1,1)
    l1_out = layer1.forward([[1, 1]])
    a1_out = act.forward(l1_out)
    l2_out = layer2.forward(a1_out)
    smax_out = smax.forward(l2_out)
    predicted[0] = np.argmax(smax_out)
    print('softmax out for (1,1)', smax_out, 'predicted label:', int(predicted[0]))

    # predicting label for (0,0)
    l1_out = layer1.forward([[0, 0]])
    a1_out = act.forward(l1_out)
    l2_out = layer2.forward(a1_out)
    smax_out = smax.forward(l2_out)
    predicted[1] = np.argmax(smax_out)
    print('softmax out for (0,0)', smax_out, 'predicted label:', int(predicted[1]))

    # predicting label for (1,0)
    l1_out = layer1.forward([[1, 0]])
    a1_out = act.forward(l1_out)
    l2_out = layer2.forward(a1_out)
    smax_out = smax.forward(l2_out)
    predicted[2] = np.argmax(smax_out)
    print('softmax out for (1,0)', smax_out, 'predicted label:', int(predicted[2]))

    # predicting label for (0,1)
    l1_out = layer1.forward([[0, 1]])
    a1_out = act.forward(l1_out)
    l2_out = layer2.forward(a1_out)
    smax_out = smax.forward(l2_out)
    predicted[3] = np.argmax(smax_out)
    print('softmax out for (0,1)', smax_out, 'predicted label:', int(predicted[3]))

    print('total predicted labels:', predicted.astype('uint8'))

    accuracy += (predicted[0] == 0) & (predicted[1] == 0) & (predicted[2] == 1) & (predicted[3] == 1)

    if (j + 1) % 10 == 0:
        print('test iteration:', j + 1)

print('accuracy:', accuracy / num_test * 100, '%')