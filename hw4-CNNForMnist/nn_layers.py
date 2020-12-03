import numpy as np
from skimage.util.shape import view_as_windows

##########
#   convolutional layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_convolutional_layer:

    def __init__(self, Wx_size, Wy_size, input_size, in_ch_size, out_ch_size, std=1e0):
    
        # initialization of weights
        self.W = np.random.normal(0, std / np.sqrt(in_ch_size * Wx_size * Wy_size / 2),
                                  (out_ch_size, in_ch_size, Wx_size, Wy_size))
        self.b = 0.01 + np.zeros((1, out_ch_size, 1, 1))
        self.input_size = input_size

    def update_weights(self, dW, db):
        self.W += dW
        self.b += db

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########
        xshape = x.shape
        Wshape = self.W.shape
        outputR = xshape[2] - Wshape[2] + 1
        outputC = xshape[3] - Wshape[3] + 1

        # (batch_size, num_filters, 26, 26)
        out = np.zeros((xshape[0], Wshape[0], outputR, outputC))

        for i in range(xshape[0]):          # batch_size
            for f in range(Wshape[0]):      # num_filters
                results = np.zeros((outputR, outputC))
                for j in range(xshape[1]):  # in_ch_size, depth
                    y = view_as_windows(x[i][j], (Wshape[2], Wshape[3]))
                    y = y.reshape((outputR, outputC, -1))
                    result = y.dot(self.W[f][j].reshape(-1, 1)) # result.shape = (30, 30)
                    results += np.squeeze(result, axis=2)

                # results是这个bitch的这个filter加完之后的值
                out[i][f] = results + self.b[0][f]

        return out


    def backprop(self, x, dLdy):

        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########
        xshape = x.shape
        # dLdyshape = (batch_size, num_filter, out_width, out_height)   size = 26
        dLdyshape = dLdy.shape
        dLdWshape = self.W.shape
        outputR = xshape[2] - dLdyshape[2] + 1
        outputC = xshape[3] - dLdyshape[3] + 1


        # dLdb
        dLdb = np.zeros((1, dLdyshape[1], 1, 1))
        for i in range(dLdyshape[1]):   # num_filters
            gradientSum = 0
            for j in range(xshape[0]):  # batch_size
                gradientSum += np.sum(dLdy[j][i])
            # add 8 batches ande divide (batch size * out_width * out_height)
            #dLdb[0][i][0][0] = gradientSum / (xshape[0] * dLdyshape[2] * dLdyshape[3])
            #dLdb[0][i][0][0] = gradientSum / (xshape[0])
            dLdb[0][i][0][0] = gradientSum

        # dLdW  # dLdW.shape = (8, 3, 3, 3)  # convolution of x and dLdy
        dLdW = np.zeros(self.W.shape)
        for f in range(dLdyshape[1]):   # filter number
            for i in range(xshape[1]):  # depth
                resultSum = np.zeros((outputR, outputC))
                #for j in random.sample(range(1, xshape[0]), M):  # batch size
                for j in range(xshape[0]):
                    y = view_as_windows(x[j][i], (dLdyshape[2], dLdyshape[3]))
                    y = y.reshape((outputR, outputC, -1))
                    result = y.dot(dLdy[j][f].reshape(-1, 1))
                    resultSum += np.squeeze(result, axis=2) # 8个batches的梯度和
                #dLdW[f][i] = np.sum(resultSum, keepdims=True, axis=0) / (dLdWshape[2] * dLdWshape[3] * xshape[0])
                #dLdW[f][i] = resultSum / (xshape[0])
                dLdW[f][i] = resultSum

        # dLdx  # dLdx.shape = (8, 3, 32, 32)
        dLdx = np.zeros(xshape)
        for i in range(xshape[0]):      # batch size
            for j in range(xshape[1]):  # depth
                resultSum = np.zeros((xshape[2], xshape[3]))
                for f in range(dLdWshape[0]):   # filter number
                    dLdyPad = self.matrixPad(dLdy[i][f], dLdWshape[3]-1, dLdWshape[2]-1)  # dLdy zero padding
                    #WFlip = matrixFlip(self.W[f][j])         # flip of W
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


    def matrixFlip(self, x):
        ret = x.reshape(x.size)
        ret = ret[::-1]
        ret = ret.reshape(x.shape)
        return ret

    def matrixPad(self, x, Rpad, Cpad):
        return np.pad(x, ((Rpad, Rpad), (Cpad, Cpad)), constant_values = (0, 0))


##########
#   max pooling layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_max_pooling_layer:
    def __init__(self, stride, pool_size):
        self.stride = stride
        self.pool_size = pool_size

    def forward(self, x):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########
        xshape = x.shape
        # out.shape = (8, 3, 16, 16)
        out = np.zeros((xshape[0], xshape[1], int((xshape[2]-self.pool_size)/self.stride+1), int((xshape[3]-self.pool_size)/self.stride+1)))
        for i in range(xshape[0]):  # batch size
            for j in range(xshape[1]):  # depth
                out[i][j] = self.maxPoolOp(x[i][j], self.stride, self.pool_size)

        return out


    def backprop(self, x, dLdy):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########
        dLdx = np.zeros(x.shape)
        xshape = x.shape
        for i in range(xshape[0]):  # batch size
            for j in range(xshape[1]):  # depth
                dLdx[i][j] = self.maxPoolback(x[i][j], dLdy[i][j], self.stride, self.pool_size)

        return dLdx


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

        out = np.zeros((xshape[0], xshape[1]))  #
        hStep = int(xshape[0] / stride)   #
        vStep = int(xshape[1] / stride)   #

        for i in range(hStep):
            outHStack = ySplice[i][0]
            for j in range(1, vStep):
                outHStack = np.hstack((outHStack, ySplice[i][j]))
            out[i*stride:i*stride+stride] = outHStack

        return out



##########
#   fully connected layer
##########
# fully connected linear layer.
# parameters: weight matrix matrix W and bias b
# forward computation of y=Wx+b
# for (input_size)-dimensional input vector, outputs (output_size)-dimensional vector
# x can come in batches, so the shape of y is (batch_size, output_size)
# W has shape (output_size, input_size), and b has shape (output_size,)

class nn_fc_layer:

    def __init__(self, input_size, output_size, std=1):
        # Xavier/He init
        self.W = np.random.normal(0, std/np.sqrt(input_size/2), (output_size, input_size))
        self.b=0.01+np.zeros((output_size))

    def forward(self,x):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########
        xreshape = x.reshape(x.shape[0], -1)
        out = xreshape @ self.W.T + self.b

        return out

    def backprop(self,x,dLdy):

        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########
        dLdb = np.squeeze(np.sum(dLdy, axis=0, keepdims=True))
        dLdW = dLdy.T @ x
        dLdx = dLdy @ self.W

        return dLdx,dLdW,dLdb

    def update_weights(self,dLdW,dLdb):

        # parameter update
        self.W += dLdW
        self.b += dLdb

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

##########
#   activation layer
##########
#   This is ReLU activation layer.
##########

class nn_activation_layer:
    
    # performs ReLU activation
    def __init__(self):
        pass
    
    def forward(self, x):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########
        out = x.copy()
        out[out<0] = 0
        
        return out
    
    def backprop(self, x, dLdy):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########
        relu_mask = (x>0)
        dLdx = dLdy * relu_mask
        
        return dLdx


##########
#   softmax layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_softmax_layer:

    def __init__(self):
        pass

    def forward(self, x):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########
        s_exp = np.exp(x - np.max(x, axis=1).reshape(x.shape[0], -1))
        s_sum = np.sum(s_exp, axis=1, keepdims=True)
        out = s_exp / s_sum

        return out

    def backprop(self, x, dLdy):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########

        return dLdx

##########
#   cross entropy layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_cross_entropy_layer:

    def __init__(self):
        pass

    def forward(self, x, y):

        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########
        CEloss = 0
        n = x.shape[0]
        for index in range(n):
            #temp = x[y[index]][index]  # 表示第index个sc的label的分数是多少
            temp = x[index][y[index]]
            if temp == 0:
                temp = 1e-8
            CEloss += -np.log10(temp)

        return CEloss / n


    def backprop(self, x, y):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########
        gradient = x.copy()
        ylen = len(y)
        for i in range(ylen):
            gradient[i][y[i]] -= 1

        return gradient / ylen
