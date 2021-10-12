

## 用pytorch跑cifar10数据集说明

> Author ： huchenlu

#### 环境要求

* Windows

* python（3.6 +）

#### 依赖包

> pytorch、numpy、matplotlib

#### 数据来源

[cifar10数据集](http://www.cs.toronto.edu/~kriz/cifar.html)

该数据集包含60000张32x32的图像，分成10个类，每个类6000张图像。这个数据集按照5:1进行划分，50000张图像相关作为训练集，10000张图像作为测试集。测试集从10个类中随机挑选，每类挑选1000张图，剩下的图像就作为训练集。

![举例](https://github.com/kelseyscat/desktop-tutorial/blob/main/pics/1.png)

#### 数据加载和处理

![image-20211012161408930](https://github.com/kelseyscat/desktop-tutorial/blob/main/pics/2.png)

* ToTensor**是指把PIL.Image(RGB) 或者numpy.ndarray(H x W x C) 从0到255的值映射到0到1的范围内，并转化成Tensor格式。

  这是归一化的过程，目的是消除特征的差异。

* Normalize(mean，std)**是实现标准化，公式是channel=（channel-mean）/std，让整体数据由一般的正态分布变化到N（0，1）

  标准化后，实现了数据中心化，这符合数据分布规律，能增加模型的泛化能力。

![image-20211012161540204](https://github.com/kelseyscat/desktop-tutorial/blob/main/pics/3.png)

* cifar10的数据已经被封装在torchvision.datasets中了，这里就是把它下载下来，如果下载训练集train = True，下载测试集就设置为False，transform对数据进行变换。

![image-20211012161951006](https://github.com/kelseyscat/desktop-tutorial/blob/main/pics/4.png)

* trainset是大数据量，要用DataLoade对其进行 shuffle, 分割成mini-batch
* pytorch中dataloader的大小将根据batch_size的大小自动调整。如果训练数据集有1000个样本，并且batch_size的大小为10，则dataloader的长度就是100。
* shuffle表示是否要把数据打乱，num_workers表示可以有多少个并行的进程（ps：我是在windows用cpu跑的，num_workers只能设为0）

#### 模型建立

首先先了解一下卷积的基础知识：

一般**CNN的架构**是这样的：input一张图像，先通过Convolution的layer（卷积层），接下来做Max pooling（池化），然后再做Convolution，再做Max pooling...

这个process可以反复进行多次，次数事先决定。最后把数据进行flatten（展成一列），把flatten output丢到一般的全连接层里去，最终得到影像识别结果。

![image-20211012162342811](https://github.com/kelseyscat/desktop-tutorial/blob/main/pics/5.png)

* **nn.Module是所有神经网络的基类**，我们自己定义任何神经网络， **都要继承nn.Module**

* **Conv2d(in_channels, out_channels, kernel_size, stride=1,padding=0, dilation=1, groups=1,bias=True, padding_mode=‘zeros’)**

  举个例子，Conv2d(1, 20, 5)，输入是**1**通道的图像，输出是**20**通道，也就是**20**个卷积核，卷积核是**5*****5**。

  **conv2**的输入：四维，常用于图像卷积，[batch, channels, H, W]。

  print了一下输入数据的shape，![img](https://github.com/kelseyscat/desktop-tutorial/blob/main/pics/6.png)，图像格式是这样的，4个batch，3个channels，长和宽都是32像素。

  所以Conv2d(3, 6, 5)表示输入的是3通道的图像，输出6通道的图像，而一般卷积层采用的都是5x5大小的卷积核。

* **torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)**

  c池化层一般就设置kernel_size和stride（步长）

* **全连接层**就是把以前的局部特征重新通过权值矩阵组装成完整的图。因为用到了所有的局部特征，所以叫全连接。

* **x = x.view()**

   可以将四维张量转化为二维张量，只有这样才能作为全连接层的输入

#### 定义损失函数和优化器

![image-20211012164945131](https://github.com/kelseyscat/desktop-tutorial/blob/main/pics/7.png)

* 用交叉熵作为损失函数，并且采用随机梯度下降的方法对梯度进行优化
* pytorch将深度学习中常用的优化方法全部封装在torch.optim之中

#### 训练

![image-20211012165456573](https://github.com/kelseyscat/desktop-tutorial/blob/main/pics/8.png)

* 想要计算各个variable的梯度，只需调用根节点的backward方法，Autograd就会自动沿着整个计算图进行反向计算，而loss就是根节点。

* **loss.backward()**代码就是在实现反向传播，自动计算所有的梯度

* 步骤：

  ①把trainloader里的数据给转换成variable，作为网络的输入

  ②每次循环新开始时，要确保梯度归零

  ③forward+backward，就是调用net()实现前传，loss.backward()实现后传。每结束一次循环，要确保梯度更新

* 训练过程

  ![image-20211012170554273](https://github.com/kelseyscat/desktop-tutorial/blob/main/pics/9.png)

#### 测试

* 先是随机取4张图片进行测试

  ![image-20211012170510683](https://github.com/kelseyscat/desktop-tutorial/blob/main/pics/10.png)

  结果为

  ![image-20211012170728378](https://github.com/kelseyscat/desktop-tutorial/blob/main/pics/11.png)

* 接下来看一下整体的正确率

  ![image-20211012170834598](https://github.com/kelseyscat/desktop-tutorial/blob/main/pics/12.png)

  结果为

  ![image-20211012170900138](https://github.com/kelseyscat/desktop-tutorial/blob/main/pics/13.png)

  正确率在53%左右

* 看一下每个类分类正确的概率

  ![image-20211012171305240](https://github.com/kelseyscat/desktop-tutorial/blob/main/pics/14.png)

  结果为

  ![image-20211012171520815](https://github.com/kelseyscat/desktop-tutorial/blob/main/pics/15.png)

#### 特别说明

本例主要参考https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#load-and-normalize-cifar10，是一个非常适合深度学习入门的案例。

  
