# ------------------------------------------
# Author: 임준상
# 
# Date: Nov 29, 2020
# ------------------------------------------

"""
Please use search(ctrl+f) to find the parameter.

Switch parameter:
    load_para:      If True, then load existing W and b.
                    Otherwise, use randomly generated W and b.

    is_learning:    If True, then training the model.

    If you just test the accuracy of image recognition,
    set the "load_para = True" and "is_learning = False".

    If you want training model then predict image value,
    set the "load_para = False" and "is_learning = True".
    (It will take a lot of time)

This model is base on Mini-SGD + 1/t decay + RMSProp

"""

import numpy as np
import os
import urllib.request
import gzip
import matplotlib.pyplot as plt
from skimage.util.shape import view_as_windows
import random

def load_mnist():
    url_tr_dat = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
    url_tr_lab = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
    url_ts_dat = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
    url_ts_lab = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'

    if not os.path.exists('./mnist-batches-py'):
        os.mkdir('mnist-batches-py')

        urllib.request.urlretrieve(url_tr_dat, './mnist-batches-py/train-images-idx3-ubyte.gz')
        urllib.request.urlretrieve(url_tr_lab, './mnist-batches-py/train-labels-idx1-ubyte.gz')
        urllib.request.urlretrieve(url_ts_dat, './mnist-batches-py/t10k-images-idx3-ubyte.gz')
        urllib.request.urlretrieve(url_ts_lab, './mnist-batches-py/t10k-labels-idx1-ubyte.gz')

    X_train_f = gzip.open('./mnist-batches-py/train-images-idx3-ubyte.gz', 'rb')
    y_train_f = gzip.open('./mnist-batches-py/train-labels-idx1-ubyte.gz', 'rb')
    X_test_f = gzip.open('./mnist-batches-py/t10k-images-idx3-ubyte.gz', 'rb')
    y_test_f = gzip.open('./mnist-batches-py/t10k-labels-idx1-ubyte.gz', 'rb')

    s = X_train_f.read()
    loaded = np.frombuffer(s, dtype=np.uint8)
    X_train = loaded[16:].reshape((60000, 1, 28, 28)).astype(float)

    s = y_train_f.read()
    loaded = np.frombuffer(s, dtype=np.uint8)
    y_train = loaded[8:].reshape((60000,)).astype('uint8')

    s = X_test_f.read()
    loaded = np.frombuffer(s, dtype=np.uint8)
    X_test = loaded[16:].reshape((10000, 1, 28, 28)).astype('uint8')

    s = y_test_f.read()
    loaded = np.frombuffer(s, dtype=np.uint8)
    y_test = loaded[8:].reshape((10000,)).astype('uint8')

    X_train_f.close()
    y_train_f.close()
    X_test_f.close()
    y_test_f.close()

    return X_train, y_train, X_test, y_test

#################################################################################

class nn_convolutional_layer:

    def __init__(self, filter_width, filter_height, input_size, in_ch_size, num_filters, std=1e0):
        # initialization of weights
        self.W = np.random.normal(0, std / np.sqrt(in_ch_size * filter_width * filter_height / 2),
                                  (num_filters, in_ch_size, filter_width, filter_height))
        self.b = 0.01 + np.zeros((1, num_filters, 1, 1))
        self.input_size = input_size

        #######
        ## If necessary, you can define additional class variables here
        #######

    def update_weights(self, dW, db):
        self.W += dW
        self.b += db

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

    #######
    # Q1. Complete this method
    #######
    def forward(self, x):
        # x.shape = (8, 3, 32, 32)
        # W.shape = (8, 3, 3, 3)
        # b.shape = (1, 8, 1, 1)
        xshape = x.shape
        Wshape = self.W.shape
        outputR = xshape[2] - Wshape[2] + 1 # 30
        outputC = xshape[3] - Wshape[3] + 1 # 30
        depth = xshape[1]     # 3
        batchSize = xshape[0]   # 8
        numFilters = Wshape[0]  # 8

        out = np.zeros((batchSize, numFilters, outputR, outputC))   # out.shape = (8, 8, 30, 30)

        for i in range(batchSize):
            for f in range(numFilters):
                #results = np.zeros((depth, outputR, outputC))  # results.shape = (3, 30, 30)
                results = np.zeros((outputR, outputC))   # results.shape = (30, 30)
                for j in range(depth):
                    y = view_as_windows(x[i][j], (Wshape[2], Wshape[3]))
                    y = y.reshape((outputR, outputC, -1))
                    result = y.dot(self.W[f][j].reshape(-1, 1)) # result.shape = (30, 30)
                    results += np.squeeze(result, axis=2)

                # 此时results已经是三个depth加完之后的了
                out[i][f] = results + self.b[0][f]

        #print(out.shape)   # (8, 8, 30, 30)
        return out



    #######
    # Q2. Complete this method
    #######
    def backprop(self, x, dLdy):
        # x.shape = (8, 3, 32, 32)
        # dLdy.shape = (8, 8, 30, 30)
        # dLdx.shape = (8, 3, 32, 32)
        # dLdW.shape = (8, 3, 3, 3)
        # dLdb.shape = (1, 8, 1, 1)
        xshape = x.shape
        dLdyshape = dLdy.shape
        dLdWshape = self.W.shape
        outputR = xshape[2] - dLdyshape[2] + 1 # 3
        outputC = xshape[3] - dLdyshape[3] + 1 # 3

        # dLdb
        dLdb = np.zeros((1, dLdyshape[1], 1, 1))
        for i in range(dLdyshape[1]):   # filter number
            gradientSum = 0
            for j in range(xshape[0]):  # batch size
                gradientSum += np.sum(dLdy[j][i])
            # add 8 batches ande divide (batch size * out_width * out_height)
            #dLdb[0][i][0][0] = gradientSum / (xshape[0] * dLdyshape[2] * dLdyshape[3])
            dLdb[0][i][0][0] = gradientSum / (xshape[0])
            #dLdb[0][i][0][0] = gradientSum

        # dLdW  # dLdW.shape = (8, 3, 3, 3)  # convolution of x and dLdy
        dLdW = np.zeros(self.W.shape)
        for f in range(dLdyshape[1]):   # filter number
            for i in range(xshape[1]):  # depth
                resultSum = np.zeros((outputR, outputC))
                for j in random.sample(range(1, xshape[0]), M):  # batch size
                    y = view_as_windows(x[j][i], (dLdyshape[2], dLdyshape[3]))
                    y = y.reshape((outputR, outputC, -1))
                    result = y.dot(dLdy[j][f].reshape(-1, 1))
                    resultSum += np.squeeze(result, axis=2) # 8个batches的梯度和
                #dLdW[f][i] = np.sum(resultSum, keepdims=True, axis=0) / (dLdWshape[2] * dLdWshape[3] * xshape[0])
                dLdW[f][i] = resultSum / M
                #dLdW[f][i] = resultSum

        # dLdx  # dLdx.shape = (8, 3, 32, 32)
        dLdx = np.zeros(xshape)
        for i in range(xshape[0]):  # batch size
            for j in range(xshape[1]):  # depth
                resultSum = np.zeros((xshape[2], xshape[3]))
                for f in range(dLdWshape[0]):   # filter number
                    dLdyPad = self.matrixPad(dLdy[i][f], dLdWshape[3]-1, dLdWshape[2]-1)  # dLdy zero padding
                    #WFlip = matrixFlip(self.W[f][j])            # flip of W
                    WFlip = self.matrixFlip(self.W[f][j])
                    dLdyPadShape = dLdyPad.shape    # dLdyPad.shape = (34, 34)
                    WFlipShape = WFlip.shape        # WFlip.shape = (3, 3)
                    xOutputR = dLdyPadShape[0] - WFlipShape[0] + 1  # 32
                    xOutputC = dLdyPadShape[1] - WFlipShape[1] + 1  # 32
                    # convolution operating
                    y = view_as_windows(dLdyPad, (WFlipShape[0], WFlipShape[1]))
                    y = y.reshape((xOutputR, xOutputC, -1))
                    result = y.dot(WFlip.reshape(-1, 1))
                    resultSum += np.squeeze(result, axis=2) # 8个filters的梯度和
                dLdx[i][j] = resultSum / dLdWshape[0]
                #dLdx[i][j] = resultSum

        return dLdx, dLdW, dLdb

    #######
    ## If necessary, you can define additional class methods here
    def matrixFlip(self, x):
        ret = x.reshape(x.size)
        ret = ret[::-1]
        ret = ret.reshape(x.shape)
        return ret

    def matrixPad(self, x, Rpad, Cpad):
        return np.pad(x, ((Rpad, Rpad), (Cpad, Cpad)), constant_values = (0, 0))
    #######



class nn_max_pooling_layer:
    def __init__(self, stride, pool_size):
        self.stride = stride
        self.pool_size = pool_size
        #######
        ## If necessary, you can define additional class variables here
        #######

    #######
    # Q3. Complete this method
    #######
    def forward(self, x):
        # x.shape = (8, 3, 32, 32)
        xshape = x.shape
        # out.shape = (8, 3, 16, 16)
        out = np.zeros((xshape[0], xshape[1], int((xshape[2]-self.pool_size)/self.stride+1), int((xshape[3]-self.pool_size)/self.stride+1)))
        for i in range(xshape[0]):  # batch size
            for j in range(xshape[1]):  # depth
                out[i][j] = self.maxPoolOp(x[i][j], self.stride, self.pool_size)


        return out

    #######
    # Q4. Complete this method
    #######
    def backprop(self, x, dLdy):
        # x.shape = (8, 3, 32, 32)
        # dLdy.shape = (8, 3, 16, 16)
        dLdx = np.zeros(x.shape)
        xshape = x.shape
        for i in range(xshape[0]):  # batch size
            for j in range(xshape[1]):  # depth
                dLdx[i][j] = self.maxPoolback(x[i][j], dLdy[i][j], self.stride, self.pool_size)

        return dLdx

    #######
    ## If necessary, you can define additional class methods here
    def maxPoolOp(self, x, stride, pool_size):  # only receive 2-D np
        y = view_as_windows(x, (pool_size, pool_size), step=stride)
        outputR = int((x.shape[0]-pool_size)/stride+1)
        outputC = int((x.shape[1]-pool_size)/stride+1)
        y = y.reshape((outputR, outputC, -1))
        ret = np.max(y, keepdims=True, axis=2)
        return ret.reshape(outputR, outputC)

    def maxPoolback(self, x, dLdy, stride, pool_size):
        xshape = x.shape
        y = view_as_windows(x, (pool_size, pool_size), step=stride)
        windowR = int((x.shape[0]-pool_size)/stride+1) # 
        windowC = int((x.shape[1]-pool_size)/stride+1) # 
        y = y.reshape((windowR, windowC, -1))
        temp = np.argmax(y, axis=2)

        yback = np.zeros(y.shape)   # shape = 
        ySplice = np.zeros((y.shape[0], y.shape[1], pool_size, pool_size))
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                yback[i][j][temp[i][j]] = dLdy[i][j]    # 除了max其他全是0，argmax处为梯度
                ySplice[i][j] = yback[i][j].reshape((pool_size, pool_size))

        # shape = 
        """ySplice = np.zeros((y.shape[0], y.shape[1], pool_size, pool_size))
        #print(yback[0][0].reshape((pool_size, pool_size)))
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                ySplice[i][j] = yback[i][j].reshape((pool_size, pool_size))"""

        out = np.zeros((xshape[0], xshape[1]))  #
        hStep = int(xshape[0] / stride)   #
        vStep = int(xshape[1] / stride)   #

        for i in range(hStep):
            outHStack = ySplice[i][0]
            for j in range(1, vStep):
                outHStack = np.hstack((outHStack, ySplice[i][j]))
            out[i*stride:i*stride+stride] = outHStack

        return out



class fully_connect_layer:

    # linear layer.
    # randomly initialized by creating matrix W and bias b
    def __init__(self, input_size, output_size, std=1):
        self.W = np.random.normal(0, std, (output_size, input_size))
        self.b = np.random.normal(0, std, (output_size, 1))

    ######
    ## Q1
    def forward(self, x):
        s = self.W @ x + self.b
        return s

    ######
    ## Q2
    ## returns three parameters
    def backprop(self, x, dLdy):
        #lenc(x)        4 20
        #lenc(dLdy)  2 20
        #sizeb = np.shape(self.b)[0]
        gradient = (np.sum(dLdy, axis=1, keepdims=True))
        dLdb = gradient
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



class nn_softmax_layer:
    def __init__(self):
        pass

    ######
    ## Q5
    def forward(self, x):
        s_exp = np.exp(x - np.max(x, axis=0))
        s_sum = np.sum(s_exp, axis=0, keepdims=True)
        sc = s_exp / s_sum
        return sc

    ######
    ## Q6
    def backprop(self, x, dLdy):
        return dLdy



class nn_cross_entropy_layer:
    def __init__(self):
        pass

    def forward(self, sc, y):
        # cent
        CEloss = 0
        n = sc.shape[1]
        for index in range(n):
            temp = sc[y[index]][index]  # 表示第index个sc的label的分数是多少
            if temp == 0:
                temp = 1e-15
            CEloss += -np.log10(temp)

        return CEloss

    ######
    ## Q8
    def backprop(self, x, y):
        gradient = x
        ylen = len(y)
        for i in range(ylen):
            gradient[y[i]][i] -= 1
        return gradient / ylen



class smax_cent_layer:
    def __init__(self):
        pass

    def forward(self, x, y):
        # softmax
        s_exp = np.exp(x - np.max(x, axis=0))
        s_sum = np.sum(s_exp, axis=0, keepdims=True)
        sc = s_exp / s_sum

        # cent
        CEloss = 0
        n = sc.shape[1]
        for index in range(n):
            temp = sc[y[index]][index]  # 表示第index个sc的label的分数是多少
            if temp == 0:
                temp = 1e-15
            CEloss += -np.log10(temp)

        return CEloss

    def backprop(self, x, y):
        gradient = x
        ylen = len(y)
        yl = y.tolist()
        gradient[yl, range(ylen)] -= 1
        return gradient / ylen



#################################################################################


mnist_data = load_mnist()

X_train = mnist_data[0]
y_train = mnist_data[1]

X_test = mnist_data[2]
y_test = mnist_data[3]

# Check the size of the training and test data.
print('Training data shape: ', X_train.shape)       # (60000, 1, 28, 28)
print('Training labels shape: ', y_train.shape)     # (60000,)
print('Test data shape: ', X_test.shape)            # (10000, 1, 28, 28)
print('Test labels shape: ', y_test.shape)           # (10000,)

# switch
load_para = True
is_learning = False

#X = np.concatenate((X_train, X_test), axis=0)
#y = np.concatenate((y_train, y_test), axis=0)

X = X_train
y = y_train

#X = X_test
#y = y_test

# select three random number images
num_plot = 3
sample_index = np.random.randint(0,X.shape[0],(num_plot,))  # batch size张图中随机选三张
predicted = np.ones(num_plot)

Xshape = X.shape
yshape = y.shape

batch_size = Xshape[0]          # 图的数量
input_size = Xshape[2]          # 图大小（像素）
in_ch_size = Xshape[1]          # 图的depth (grayscale or RGB)
filter_width = 5                # filter size
filter_height = filter_width
num_filters = 25                # filter number
class_num = 10

# some para
num_train = 50                  # training times
lr = 3.0
cnv_lr = lr
fcl_lr = lr                     # learning rate
decay = 0.05
break_threshold = 400000
M = 128
cnvRMS_r_W = 0
fclRMS_r_W = 0
alpha = 0.9

# maxpools setting
mpl_stride = 4
mpl_size = 4

#
std = 1e0
dt = 1e-2

# output size
conv_out_size = input_size - filter_width + 1
mpl_out_size = int((conv_out_size - mpl_size) / mpl_stride + 1)
#print(conv_out_size, mpl_out_size)

# fully connect layer para
fcl_filter_width = mpl_out_size
fcl_filter_height = fcl_filter_width
fcl_input_size = mpl_out_size
fcl_in_ch_size = num_filters
fcl_num_filters = class_num

# print parameter
if is_learning == True:
    print('batch_size: %s, num_filters: %s' % (batch_size, num_filters))
    print('lr: %s, decay: %s, lr when 100th: %s, lr when 500th: %s' % (lr, decay, (lr/(1.0+decay*100.0)), (lr/(1.0+decay*500.0))))
    print('break_threshold: %s, SGD M: %s' % (break_threshold, M))

# function declaration
# create convolutional layer object
cnv = nn_convolutional_layer(filter_width, filter_height, input_size, in_ch_size, num_filters, std)
# max pools
mpl = nn_max_pooling_layer(stride=mpl_stride, pool_size=mpl_size)
# fully connect layer
#fcl = fully_connect_layer(in_ch_size*mpl_out_size*mpl_out_size*num_filters, class_num)
fcl = nn_convolutional_layer(fcl_filter_width, fcl_filter_height, fcl_input_size, fcl_in_ch_size, fcl_num_filters, std)
# softmax cross entropy
smax_cent = smax_cent_layer()
# softmax
smax = nn_softmax_layer()     # 接收 10-by-1的column vector
# cross entropy
cent = nn_cross_entropy_layer()
# loss
loss_out = np.zeros(num_train)

# load para
if load_para == True:
    cnv_load = np.load("cnv_para.npy", allow_pickle = True)
    fcl_load = np.load("fcl_para.npy", allow_pickle = True)
    cnv.set_weights(cnv_load[0], cnv_load[1])
    fcl.set_weights(fcl_load[0], fcl_load[1])

if is_learning == True:
    for ntrain in range(num_train):

        # convolution layer
        cnv_out = cnv.forward(X)    # shape = (batch_size, num_filters, conv_out_size, conv_out_size)
        #print(cnv_out.shape)

        # max pool layer
        mpl_out = mpl.forward(cnv_out)

        # fully connect layer
        fcl_out = fcl.forward(mpl_out)  # shape = (batch_size, in_ch_size(class_num), mpl_out_size, ~)

        # softmax layer
        smax_in = fcl_out.reshape(batch_size, class_num).T
        smax_out = smax.forward(smax_in)    # shape = (class_num, batch_size)

        # cent loss
        loss_out[ntrain] = cent.forward(smax_out, y)

        # back smax and cent layer
        b_smax_cent_out = cent.backprop(smax_out, y)    # (class_num, batch_size)

        # back fully connect layer
        b_fcl_in = b_smax_cent_out.T.reshape(batch_size, class_num, 1, 1)
        b_fcl_out, b_fcl_out_W, b_fcl_out_b = fcl.backprop(mpl_out, b_fcl_in)

        # back max pool layer
        b_mpl_out = mpl.backprop(cnv_out, b_fcl_out)

        # back convolution layer
        b_cnv_out, b_cnv_out_W, b_cnv_out_b = cnv.backprop(X, b_mpl_out)

        # RMSProp
        cnvRMS_r_W = (alpha*cnvRMS_r_W) + (1-alpha) * (b_cnv_out_W**2)
        cnvRMS_W = (cnv_lr*b_cnv_out_W) / (cnvRMS_r_W**0.5+1e-7)
        fclRMS_r_W = (alpha*fclRMS_r_W) + (1-alpha) * (b_fcl_out_W**2)
        fclRMS_r = (fcl_lr*b_fcl_out_W) / (fclRMS_r_W**0.5+1e-7)

        # update convolution layer
        cnv.update_weights(-cnvRMS_W, -b_cnv_out_b*cnv_lr)

        # update fully connect layer
        fcl.update_weights(-fclRMS_r, -b_fcl_out_b*fcl_lr)

        # show info
        print()
        print("[%s]th epoch(s)\nloss: %s" % (ntrain, loss_out[ntrain]))
        print("Cnv update: weights = %s, bias = %s" % (b_cnv_out_W[0][0].reshape(filter_width**2)[13:16]*cnv_lr, b_cnv_out_b[0][1:4].T*cnv_lr))
        cnv_current_para = cnv.get_weights()
        print("Cnv current para: weights =", cnv_current_para[0][0][0].reshape(filter_width**2)[13:16], ", bias =", cnv_current_para[1][0][1:4].T)

        if ntrain > 10:
            if loss_out[ntrain-1]+loss_out[ntrain-2]+loss_out[ntrain-3] < break_threshold:
                break

        # 1/t decay
        cnv_lr = lr * 1.0 / (1.0+decay*ntrain)
        fcl_lr = cnv_lr

        # save
        np.save("cnv_para_from_hw4.npy", cnv.get_weights())
        np.save("fcl_para_from_hw4.npy", fcl.get_weights())

# predict
batch_size = 1
for i in range(num_plot):
    #pred_cnv_in = X_train[sample_index[i]].reshape(1, in_ch_size, input_size, input_size)
    pred_cnv_in = X[sample_index[i]].reshape(1, in_ch_size, input_size, input_size)
    pred_cnv_out = cnv.forward(pred_cnv_in)

    pred_mpl_out = mpl.forward(pred_cnv_out)

    pred_fcl_out = fcl.forward(pred_mpl_out)

    pred_smax_in = pred_fcl_out.reshape(batch_size, class_num).T
    pred_smax_out = smax.forward(pred_smax_in)

    predicted[i] = np.argmax(pred_smax_out)



# plot the selected images
for i in range(num_plot):
    img = np.squeeze(X_train[sample_index[i]])      # 选中图片的重组和, img.shape = (1, 28, 28) --> (28, 28)
    #img = np.squeeze(X[sample_index[i]])
    ax = plt.subplot('1'+str(num_plot)+str(i))
    plt.imshow(img,cmap=plt.get_cmap('gray'))
    ######
    ## Q5. Complete the below function ax.set_title
    tt = predicted[i]
    #####
    ax.set_title(predicted[i])

plt.show()