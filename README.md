# Introduction
* 这是大学课程深度学习(deep learning)的作业备份。  
* 没有使用tensorflow, pytorch等方法而是主要基于numpy的自己计算来实现。毕竟只是大学生，通过从头到尾自己实现的过程先学习和掌握原理。  
* 具体的内容在文件夹里有README介绍，结合作业要求文件和代码文件一起查看能更好理解。

# 各个文件夹的介绍
* hw1里是**knn, softmax, cross entropy**的实现，主要是为了后面两个作业（hw2, hw3）做铺垫，因为后面会用到。  
* hw2里是**反向传播神经网络(backpropagation neural network)**。由input layer，一个hidden layer，以及output layer组成。具体情况可参考hw2中的作业说明文件（英文）以及README。  
* hw3里是**卷积神经网络(convolutional neural network)**，以下用CNN代称。这个作业出了大乌龙😑，本来教授的意思是到hw4的时候再让我们写整个的CNN模型，但是我错误地理解了意思所以我在hw3里就完全自己愣头青自己看材料写了个CNN模型。
* hw4里是**基于卷积神经网络的mnist**。hw4跟hw3不同的地方在于
	* hw3是我自己写得CNN模型，框架很简单，没有激活函数(activation function)，也没有设置**训练准确度**和**验证准确度**，只是简单粗暴地把训练集放进去训练。  
	* hw4的architecture是教授给的，学生只需要给出每个layers的forward和backprop，以及momentum算法就好了。教授的模型学得快准确率也高
	* 虽然hw3出了乌龙，但是通过完全自己设计和实现CNN的过程对CNN更了解了，写hw4的时候得心应手，不亏。