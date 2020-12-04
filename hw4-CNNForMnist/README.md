# Introduction
这是本学期第四次编程作业，由卷积神经网络实现的数字图片识别。  
hw3也是数字图片识别，区别在于上一次是完全我作为初学者自己写得神经网络。  
这次是教授给的框架，我只用把forward和backprop函数和momentum算法写好就行了。所以这次的模型速度又快准确率又高。  

## 一些截图  
**CNN architecture**  
<div  align="center">    
    <img src="./CNN-architecture.png" alt="CNN architecture" align=center />
</div>  

**Input**: 7万张28x28x1像素图片（六万张训练，一万张测试）  
<div  align="center">  
    <img src="./input_sample.png" alt="input image" align=center />
</div>  

**运行时**: 每**50**次update打印一次loss，每**200**次update**训练准确度**和**验证准确度**。  
* 训练准确度(training accuracy)指当前模型对接下来一批(batch)输入的准确度。  
* 验证准确度(validation accuracy)指当前模型对从训练集以外随机抽取的一批数据集的准确度。  

<div  align="center">  
    <img src="./运行时截图1.png" alt="运行时截图1" align=center width="450" height="450" />
</div>  

**Output:**  
未完待续 太困了