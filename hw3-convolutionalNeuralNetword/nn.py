import numpy as np
from skimage.util.shape import view_as_windows


#######
# if necessary, you can define additional functions which help your implementation,
# and import proper libraries for those functions.
#######

class nn_convolutional_layer:

    def __init__(self, filter_width, filter_height, input_size, in_ch_size, num_filters, std=1e0):
        # initialization of weights
        # W.shape = (8, 3, 3, 3)
        # b.shape = (1, 8, 1, 1)
        # input_size = 32
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
        depth = xshape[1]       # 3
        batchSize = xshape[0]      # 8
        numFilters = Wshape[0]  # 8

        out = np.zeros((batchSize, numFilters, outputR, outputC))   # out.shape = (8, 8, 30, 30)

        for i in range(batchSize):
            for f in range(numFilters):
                #results = np.zeros((depth, outputR, outputC))   # results.shape = (3, 30, 30)
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
            #dLdb[0][i][0][0] = gradientSum / (xshape[0])
            dLdb[0][i][0][0] = gradientSum

        # dLdW  # dLdW.shape = (8, 3, 3, 3)  # convolution of x and dLdy
        dLdW = np.zeros(self.W.shape)
        for f in range(dLdyshape[1]):    # filter number
            for i in range(xshape[1]):  # depth
                resultSum = np.zeros((outputR, outputC))
                for j in range(xshape[0]):  # batch size
                    y = view_as_windows(x[j][i], (dLdyshape[2], dLdyshape[3]))
                    y = y.reshape((outputR, outputC, -1))
                    result = y.dot(dLdy[j][f].reshape(-1, 1))
                    resultSum += np.squeeze(result, axis=2) # 8个batches的梯度和
                #dLdW[f][i] = np.sum(resultSum, keepdims=True, axis=0) / (dLdWshape[2] * dLdWshape[3] * xshape[0])
                #dLdW[f][i] = resultSum / (xshape[0])
                dLdW[f][i] = resultSum

        # dLdx  # dLdx.shape = (8, 3, 32, 32)
        dLdx = np.zeros(xshape)
        for i in range(xshape[0]):  # batch size
            for j in range(xshape[1]):  # depth
                resultSum = np.zeros((xshape[2], xshape[3]))
                for f in range(dLdWshape[0]):   # filter number
                    dLdyPad = self.matrixPad(dLdy[i][f], dLdWshape[3]-1, dLdWshape[2]-1)  # dLdy zero padding
                    #WFlip = matrixFlip(self.W[f][j])                     # flip of W
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
                #dLdx[i][j] = resultSum / dLdWshape[0]
                dLdx[i][j] = resultSum

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
    def maxPoolOp(self, x, stride, pool_size):    # only receive 2-D np
        y = view_as_windows(x, (pool_size, pool_size), step=2)
        outputR = int((x.shape[0]-pool_size)/2.0+1)
        outputC = int((x.shape[1]-pool_size)/2.0+1)
        y = y.reshape((outputR, outputC, -1))
        ret = np.max(y, keepdims=True, axis=2)
        return ret.reshape(outputR, outputC)

    def maxPoolback(self, x, dLdy, stride, pool_size):
        xshape = x.shape
        y = view_as_windows(x, (pool_size, pool_size), step=2)
        windowR = int((x.shape[0]-pool_size)/2.0+1) # 
        windowC = int((x.shape[1]-pool_size)/2.0+1) # 
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
    #######


# testing the implementation

# data sizes
batch_size = 8
input_size = 32
filter_width = 3
filter_height = filter_width
in_ch_size = 3
num_filters = 8

std = 1e0
dt = 1e-3

# number of test loops
num_test = 20

# error parameters
err_dLdb = 0
err_dLdx = 0
err_dLdW = 0
err_dLdx_pool = 0

for i in range(num_test):
    # create convolutional layer object
    cnv = nn_convolutional_layer(filter_width, filter_height, input_size, in_ch_size, num_filters, std)

    # x.shape = delta.shape = (8, 3, 32, 32)
    x = np.random.normal(0, 1, (batch_size, in_ch_size, input_size, input_size))
    delta = np.random.normal(0, 1, (batch_size, in_ch_size, input_size, input_size)) * dt

    # dLdx test
    print('dLdx test')
    y1 = cnv.forward(x)
    y2 = cnv.forward(x + delta)

    bp, _, _ = cnv.backprop(x, np.ones(y1.shape))

    exact_dx = np.sum(y2 - y1) / dt
    apprx_dx = np.sum(delta * bp) / dt
    print('exact change', exact_dx)
    print('apprx change', apprx_dx)

    err_dLdx += abs((apprx_dx - exact_dx) / exact_dx) / num_test * 100

    # dLdW test
    print('dLdW test')
    W, b = cnv.get_weights()
    dW = np.random.normal(0, 1, W.shape) * dt
    db = np.zeros(b.shape)

    z1 = cnv.forward(x)
    _, bpw, _ = cnv.backprop(x, np.ones(z1.shape))
    cnv.update_weights(dW, db)
    z2 = cnv.forward(x)

    exact_dW = np.sum(z2 - z1) / dt
    apprx_dW = np.sum(dW * bpw) / dt
    print('exact change', exact_dW)
    print('apprx change', apprx_dW)

    err_dLdW += abs((apprx_dW - exact_dW) / exact_dW) / num_test * 100

    # dLdb test
    print('dLdb test')

    W, b = cnv.get_weights()

    dW = np.zeros(W.shape)
    db = np.random.normal(0, 1, b.shape) * dt

    z1 = cnv.forward(x)

    V = np.random.normal(0, 1, z1.shape)

    _, _, bpb = cnv.backprop(x, V)

    cnv.update_weights(dW, db)
    z2 = cnv.forward(x)

    exact_db = np.sum(V * (z2 - z1) / dt)
    apprx_db = np.sum(db * bpb) / dt

    print('exact change', exact_db)
    print('apprx change', apprx_db)
    err_dLdb += abs((apprx_db - exact_db) / exact_db) / num_test * 100

    # max pooling test
    # parameters for max pooling
    stride = 2
    pool_size = 2

    mpl = nn_max_pooling_layer(stride=stride, pool_size=pool_size)

    x = np.arange(batch_size * in_ch_size * input_size * input_size).reshape(
        (batch_size, in_ch_size, input_size, input_size)) + 1
    delta = np.random.normal(0, 1, (batch_size, in_ch_size, input_size, input_size)) * dt

    print('dLdx test for pooling')
    y1 = mpl.forward(x)
    dLdy = np.random.normal(0, 10, y1.shape)
    bpm = mpl.backprop(x, dLdy)

    y2 = mpl.forward(x + delta)

    exact_dx_pool = np.sum(dLdy * (y2 - y1)) / dt
    apprx_dx_pool = np.sum(delta * bpm) / dt
    print('exact change', exact_dx_pool)
    print('apprx change', apprx_dx_pool)

    err_dLdx_pool += abs((apprx_dx_pool - exact_dx_pool) / exact_dx_pool) / num_test * 100

# reporting accuracy results.
print('accuracy results')
print('conv layer dLdx', 100 - err_dLdx, '%')
print('conv layer dLdW', 100 - err_dLdW, '%')
print('conv layer dLdb', 100 - err_dLdb, '%')
print('maxpool layer dLdx', 100 - err_dLdx_pool, '%')