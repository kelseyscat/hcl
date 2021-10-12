# 首先是调用Variable、 torch.nn、torch.nn.functional
from torch.autograd import Variable  # 这一步还没有显式用到variable，但是现在写在这里也没问题，后面会用到
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim # 导入torch.potim模块
#  首先当然肯定要导入torch和torchvision，至于第三个是用于进行数据预处理的模块
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# 由于torchvision的datasets的输出是[0,1]的PILImage，所以我们先先归一化为[-1,1]的Tensor
#  首先定义了一个变换transform，利用的是上面提到的transforms模块中的Compose( )
#  把多个变换组合在一起，可以看到这里面组合了ToTensor和Normalize这两个变换

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])

# 前面的（0.5，0.5，0.5） 是 R G B 三个通道上的均值， 后面(0.5, 0.5, 0.5) 是三个通道的标准差，
# 注意通道顺序是 R G B
# 这两个tuple数据是用来对RGB 图像做归一化的

# 定义了我们的训练集，名字就叫trainset，至于后面这一堆，其实就是一个类：
# torchvision.datasets.CIFAR10( )也是封装好了的，就在我前面提到的torchvision.datasets

trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True,
                                        download=True, transform=transform)

# trainloader其实是一个比较重要的东西，我们后面就是通过trainloader把数据传入网
# 络，当然这里的trainloader其实是个变量名，可以随便取，重点是他是由后面的
# torch.utils.data.DataLoader()定义的，这个东西来源于torch.utils.data模块，

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

# 对于测试集的操作和训练集一样
testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)
# 类别信息也是需要我们给定的
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):   # 我们定义网络时一般是继承的torch.nn.Module创建新的子类
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 添加第一个卷积层
        self.pool = nn.MaxPool2d(2, 2)  # 池化层
        self.conv2 = nn.Conv2d(6, 16, 5)  # 同样是卷积层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 接着三个全连接层
        self.fc2 = nn.Linear(120, 84)  # 采用的激活函数为tanh，使用了84个神经元
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):  # 这里定义前向传播的方法
        x = self.pool(F.relu(self.conv1(x)))  # F是torch.nn.functional的别名，这里调用了relu函数 F.relu()
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

        # 和python中一样，类定义完之后实例化就很简单了，我们这里就实例化了一个net
net = Net()
criterion = nn.CrossEntropyLoss()    #同样是用到了神经网络工具箱 nn 中的交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)   #optim模块中的SGD梯度优化方式---随机梯度下降

# -------------- 训练 ----------------
for epoch in range(2):  #指定训练一共要循环几个epoch

    running_loss = 0.0  # 定义一个变量方便我们对loss进行输出
    for i, data in enumerate(trainloader, 0):  # 这里我们遇到了第一步中出现的trailoader，代码传入数据
        # enumerate是python的内置函数，既获得索引也获得数据
        inputs, labels = data  # data是从enumerate返回的data，包含数据和标签信息，分别赋值给inputs和labels
        inputs, labels = Variable(inputs), Variable(labels)  # 将数据转换成Variable
        optimizer.zero_grad()  # 要把梯度重新归零，因为反向传播过程中梯度会累加上一次循环的梯度
        outputs = net(inputs)  # 把数据输进网络net
        loss = criterion(outputs, labels)  # 计算损失值
        loss.backward()  # loss进行反向传播
        optimizer.step()  # 当执行反向传播之后，把优化器的参数进行更新，以便进行下一轮
        running_loss += loss.item()
        if i % 2000 == 1999:  # 每2000次打印一次，用running_loss进行累加
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))  # 然后再除以2000，就得到这两千次的平均损失值
            running_loss = 0.0  # 这一个2000次结束后，就把running_loss归零，下一个2000次继续使用

print('Finished Training')

dataiter = iter(testloader)  # 创建一个python迭代器，读入testloader
images, labels = dataiter.next()  # 返回一个batch_size的图片
# print images
imshow(torchvision.utils.make_grid(images))  # 展示这四张图片
print('GroundTruth: ',
      ' '.join('%5s' % classes[labels[j]] for j in range(4)))  # python字符串格式化 ' '.join表示用空格来连接后面的字符串

outputs = net(Variable(images))  # 注意这里的images是我们从上面获得的那四张图片，所以首先要转化成variable
_, predicted = torch.max(outputs.data, 1)
# 这个 _ , predicted是python的一种常用的写法，表示后面的函数其实会返回两个值
# 但是我们对第一个值不感兴趣，就写个_在那里，把它赋值给_就好，我们只关心第二个值predicted

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))  # python的字符串格式化

correct = 0  # 定义预测正确的图片数，初始化为0
total = 0  # 总共参与测试的图片数，也初始化为0
for data in testloader:  # 循环每一个batch
    images, labels = data
    outputs = net(Variable(images))  # 输入网络进行测试
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)  # 更新测试图片的数量
    correct += (predicted == labels).sum()  # 更新正确分类的图片的数量

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))  # 最后打印结果

# 看一下每个类的预测
correct_pred = {classname: 0 for classname in classes}  # 定义预测正确的图片数，初始化为0
total_pred = {classname: 0 for classname in classes}  # 定义总共参与测试的图片数，初始化为0

# again no gradients needed
with torch.no_grad():
    for data in testloader:  # 循环每一个batch
        images, labels = data
        outputs = net(images)  # 输入网络进行测试
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1  # 更新测试图片的数量
            total_pred[classes[label]] += 1  # 更新正确分类的图片的数量

# 打印结果
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                   accuracy))