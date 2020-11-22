import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import scipy.special as sp
import time
from scipy.optimize import minimize

import data_generator as dg

# you can define/use whatever functions to implememt

########################################
# Part 1. cross entropy loss
########################################
def cross_entropy_softmax_loss(Wb, x, y, num_class, n, feat_dim):
    # implement your function here
    # return cross entropy loss
    Wb = np.reshape(Wb, (-1, 1))
    b = Wb[-num_class:]
    W = np.reshape(Wb[range(num_class * feat_dim)], (num_class, feat_dim))
    x = np.reshape(x.T, (-1, n))
    s = W @ x + b

    s_exp = np.exp(s)
    s_sum = np.sum(s_exp, axis=0, keepdims=True)
    sc = s_exp / s_sum

    CEloss = 0
    for index in range(n):
        CEloss += -np.log10(sc[y[index]][index])
    return CEloss / n

########################################
# Part 2. SVM loss calculation
########################################
def svm_loss(Wb, x, y, num_class, n, feat_dim):
    # implement your function here
    # return SVM loss

    Wb = np.reshape(Wb, (-1, 1))
    b = Wb[-num_class:]
    W = np.reshape(Wb[range(num_class * feat_dim)], (num_class, feat_dim))
    x = np.reshape(x.T, (-1, n))

    s = W @ x + b

    SVMloss = []
    for index in range(n):
        loss = 0
        for i in range(num_class):
            if i == y[index]:
                continue
            loss += max(0, s[i][index] - s[y[index]][index] + 1)
        SVMloss.append(loss)

    lamb = 0.0001
    return sum(SVMloss) / n + lamb * sum(sum(W))
    #return sum(SVMloss) / n

########################################
# Part 3. kNN classification
########################################
def eucl_dis(vec1, vec2):
    return np.linalg.norm(vec1-vec2)

def knn_test(X_train, y_train, X_test, y_test, n_train_sample, n_test_sample, k):
    # implement your function here
    #return accuracy
    correctly_classified = 0

    l = []
    for i in range(n_test_sample):
        for j in range(n_train_sample):
            l.append(eucl_dis(x_train[j], x_test[i]))

    # len(d[0]) = 400, indicates distance between test0 and all of train data
    d = np.array(l).reshape(n_test_sample, n_train_sample)  # d[0]表示第0个test和400个train data的距离

    # Find the index of the three closest train data for each test
    temp = []
    for i in range(n_test_sample):
        for j in range(k):
            temp.append(np.argsort(d[i])[j])

    labels = np.array(temp).reshape(n_test_sample, k)  # 100-by-k matrix

    # 查找k个train data的class
    for n in range(n_test_sample):
        knn_train = []  # reset, 表示k个最近的train data
        for i in range(k):
            knn_train.append(y_train[labels[n][i]])

        if stats.mode(knn_train)[0] == y_test[n]:
            correctly_classified += 1
        # 少数服从多数
        # 对比y_test 对了correctly_classified数加一

    return correctly_classified / n_test_sample


# now lets test the model for linear models, that is, SVM and softmax
def linear_classifier_test(Wb, x_te, y_te, num_class,n_test):
    Wb = np.reshape(Wb, (-1, 1))
    dlen = len(x_te[0])
    b = Wb[-num_class:]
    W = np.reshape(Wb[range(num_class * dlen)], (num_class, dlen))
    accuracy = 0;

    for i in range(n_test):
        # find the linear scores
        s = W @ x_te[i].reshape((-1, 1)) + b
        # find the maximum score index
        res = np.argmax(s)
        accuracy = accuracy + (res == y_te[i]).astype('uint8')

    return accuracy / n_test

# number of classes: this can be either 3 or 4
num_class = 4

# sigma controls the degree of data scattering. Larger sigma gives larger scatter
# default is 1.0. Accuracy becomes lower with larger sigma
sigma = 1.0

print('number of classes: ',num_class,' sigma for data scatter:',sigma)
if num_class == 4:
    n_train = 400
    n_test = 100
    feat_dim = 2
else:  # then 3
    n_train = 300
    n_test = 60
    feat_dim = 2

# generate train dataset
print('generating training data')
x_train, y_train = dg.generate(number=n_train, seed=None, plot=True, num_class=num_class, sigma=sigma)

# generate test dataset
print('generating test data')
x_test, y_test = dg.generate(number=n_test, seed=None, plot=False, num_class=num_class, sigma=sigma)

# set classifiers to 'svm' to test SVM classifier
# set classifiers to 'softmax' to test softmax classifier
# set classifiers to 'knn' to test kNN classifier
classifiers = 'softmax'

if classifiers == 'svm':
    print('training SVM classifier...')
    w0 = np.random.normal(0, 1, (2 * num_class + num_class))
    result = minimize(svm_loss, w0, args=(x_train, y_train, num_class, n_train, feat_dim))
    print('testing SVM classifier...')

    Wb = result.x
    print('accuracy of SVM loss: ', linear_classifier_test(Wb, x_test, y_test, num_class,n_test)*100,'%')

elif classifiers == 'softmax':
    print('training softmax classifier...')
    w0 = np.random.normal(0, 1, (2 * num_class + num_class))
    result = minimize(cross_entropy_softmax_loss, w0, args=(x_train, y_train, num_class, n_train, feat_dim))

    print('testing softmax classifier...')

    Wb = result.x
    print('accuracy of softmax loss: ', linear_classifier_test(Wb, x_test, y_test, num_class,n_test)*100,'%')

else:  # knn
    # k value for kNN classifier. k can be either 1 or 3.
    k = 3
    print('testing kNN classifier...')
    print('accuracy of kNN loss: ', knn_test(x_train, y_train, x_test, y_test, n_train, n_test, k)*100
          , '% for k value of ', k)
