# hw3 introduction
这是主要基于python的numpy实现的卷积神经网络(convolutional neural network)，具体内容可查看作业说明PDF和代码文件~

# 项目说明
总共有两个py文件：nn.py和mnist.py  
## nn.py  
里面有nn_convolutional_layer和nn_max_pooling_layer两个classes，分别负责convolution和maxpool操作。  
每个class里主要有两个函数：forward和backward，forward负责执行操作，backward负责计算梯度。  
剩余的主体代码主要是生成数据来测试上面的两个classes是否能正常使用，以及调整各种数据后(ex: input_size, batch_size, filter_size等)代码是否依旧能运行等。  

### nn.py的运行结果  
#### batch_size设置为8时  
![image](https://github.com/Lin-CX/deep-learning/blob/main/hw3-convolutionalNeuralNetword/nnResult1.png)  
#### batch_size设置为32时(由于处理数据变多所以运行时间也变多了)  
![image](https://github.com/Lin-CX/deep-learning/blob/main/hw3-convolutionalNeuralNetword/nnResult2.png)
  
  
## mnist.py  
#### Nov 24, 2020  
各个连接层可以连接，正常调整参数(input_size, filter_num, pool_size等)。  
但是对于池化层到全连接层的过度有点不是很明白，目前是池化层结束后再将其以一维reshape。
如有4个filter，池化层结束后的大小是6-by-6，所以reshape(4×6×6)，然后再乘W得到10个labels的值。  
loss是用softmax + cross-entropy来计算的，目前问题是loss优化到一定数值后就一直在这个数字波动了。  
由于教授只讲了convolution layer和pools layer。后面的操作先看看别人的思路再进行。姑且先上传保存下进度吧。  
  
#### Nov 25, 2020
把forward和backprop的过程修改了一点，目前可以认图了。  
但是电脑配置太差，只测试了学习50，100， 500，1000，5000张图的情况，均可以正常识别出图中的数字。  
学习的图片再多的话如10000张图没试过了跑了一天都没跑完。。。但是总共有6万张图，绝望。