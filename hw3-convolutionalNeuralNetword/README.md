# hw3 introduction
这是主要基于python的numpy实现的卷积神经网络(convolutional neural network)，具体内容可查看作业说明PDF和代码文件~

# 项目说明
总共有两个py文件：nn.py和mnist.py  
<big>nn.py</big>  
里面有nn_convolutional_layer和nn_max_pooling_layer两个classes，分别负责convolution和maxpool操作。  
每个class里主要有两个函数：forward和backward，forward负责执行操作，backward负责计算梯度。  
剩余的主体代码主要是生成数据来测试上面的两个classes是否能正常使用，以及调整各种数据后(ex: input_size, batch_size, filter_size等)代码是否依旧能运行等。  

nn.py的运行结果  
batch_size设置为8时  
![image](https://github.com/Lin-CX/deep-learning/blob/main/hw3-convolutionalNeuralNetword/nnResult1.png)  
batch_size设置为32时  
![image](https://github.com/Lin-CX/deep-learning/blob/main/hw3-convolutionalNeuralNetword/nnResult2.png)
  
  
<big>mnist.py</big>  
未完待续