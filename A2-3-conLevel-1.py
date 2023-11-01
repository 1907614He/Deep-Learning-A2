#%% 数据加载和处理
import torch
import torchvision
import torchvision.transforms as transforms#用于数据预处理
import matplotlib.pyplot as plt
from torch.autograd import Variable   
import torch.nn as nn
import torch.nn.functional as F
 
#  **由于torchvision的datasets的输出是[0,1]的PILImage，所以我们先先归一化为[-1,1]的Tensor**
#  首先定义了一个变换transform，把多个变换组合在一起，这里组合了ToTensor和Normalize这两个变换

transform_form = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]) 
 
# torchvision.datasets.CIFAR10()导入数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                        transform=transform_form,download=True)
# torch.utils.data.DataLoader()定义的，这个东西来源于torch.utils.data模块
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                       transform=transform_form,download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

# 类别信息也是需要我们给定的
classes = ('plane', 'car', 'bird', 'cat',
'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#%% 定义卷积神经网络
class Net(nn.Module):                 # 我们定义网络时一般是继承的torch.nn.Module创建新的子类
    def __init__(self):    
        super(Net, self).__init__()   
        self.conv1 = nn.Conv2d(3, 6, 5)       # 添加第一个卷积层,调用了nn里面的Conv2d（通道数，卷积核数，卷积核尺寸5*5）
        self.pool = nn.MaxPool2d(2, 2)        # 最大池化层，（卷积核尺寸，步长）
        self.conv2 = nn.Conv2d(6, 16, 5)      # 同样是卷积层
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 接着三个全连接层
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
 
    def forward(self, x):           # 这里定义前向传播的方法，为什么没有定义反向传播的方法呢？这其实就涉及到torch.autograd模块了，
        x = self.pool(F.relu(self.conv1(x)))  # F是torch.nn.functional的别名，这里调用了relu函数 F.relu()
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  # .view( )是一个tensor的方法，使得tensor改变形状。
                                    #  你会发现第一个全连接层的首参数是16*5*5，所以要保证能够相乘，在矩阵乘法之前就要把x调到正确的size
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
 
    
# 类定义完之后实例化net
net = Net()
#%% 定义损失函数和优化器
import torch.optim as optim          #导入torch.potim模块
 
criterion = nn.CrossEntropyLoss()    #同样是用到了神经网络工具箱 nn 中的交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.002, momentum=0.9, weight_decay=0.0001)   #optim模块中的SGD梯度优化方式---随机梯度下降
loss_count=[]
ac_count = []
#%% 训练
for epoch in range(20):  # 指定训练一共要循环几个epoch
    running_loss = 0.0  #定义一个变量方便我们对loss进行输出
    for i, data in enumerate(trainloader, 0): # enumerate是python的内置函数，既获得索引也获得数据
        inputs, labels = data   # data是从enumerate返回的data，包含数据和标签信息，分别赋值给inputs和labels
        inputs, labels = Variable(inputs), Variable(labels) # 将数据转换成Variable，第二步里面我们已经引入这个模块
                                                           
        optimizer.zero_grad()                # 要把梯度重新归零，因为反向传播过程中梯度会累加上一次循环的梯度   
        outputs = net(inputs)                # 把数据输进网络net
        loss = criterion(outputs, labels)    # 计算损失值
        loss.backward()                      # loss进行反向传播
        optimizer.step()                     # 当执行反向传播之后，把优化器的参数进行更新，以便进行下一轮
        # print statistics                   
        running_loss += loss.item()         # 从下面一行代码可以看出它是每循环0-1999共两千次才打印一次
        if i % 2000 == 1999:    
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))  # 然后再除以2000，就得到这两千次的平均损失值
            loss_count.append(running_loss/2000)
            running_loss = 0.0               # 这一个2000次结束后，就把running_loss归零，下一个2000次继续使用
            
            correct = 0   # 定义预测正确的图片数，初始化为0
            total = 0     # 总共参与测试的图片数，也初始化为0
            for data in testloader:  # 循环每一个batch
                images, labels = data
                outputs = net(Variable(images))  # 输入网络进行测试
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)          # 更新测试图片的数量
                correct += (predicted == labels).sum() # 更新正确分类的图片的数量
            ac = 100 * correct / total
            print('Accuracy of the network on the 10000 test images: %d %%' % (ac))     # 最后打印结果
            ac_count.append(ac)
            
print('Finished Training')
#%% 测试
'''
plt.plot(loss_count, label='Training Loss')
#plt.plot(val_loss, label='Validation Loss')
plt.title('Training Loss')
plt.legend()
plt.show()
'''
correct = 0   # 定义预测正确的图片数，初始化为0
total = 0     # 总共参与测试的图片数，也初始化为0
for data in testloader:  # 循环每一个batch
    images, labels = data
    outputs = net(Variable(images))  # 输入网络进行测试
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)          # 更新测试图片的数量
    correct += (predicted == labels).sum() # 更新正确分类的图片的数量
 
print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))          # 最后打印结果
'''
plt.plot(ac_count, label='Test accuracy')
#plt.plot(val_loss, label='Validation Loss')
plt.title('Test accuracy')
plt.legend()
plt.show()
'''














#%% 定义卷积神经网络
class Net_0(nn.Module):                 # 我们定义网络时一般是继承的torch.nn.Module创建新的子类
    def __init__(self):    
        super(Net_0, self).__init__()   
        self.conv1 = nn.Conv2d(3, 6, 5)       # 添加第一个卷积层,调用了nn里面的Conv2d（通道数，卷积核数，卷积核尺寸5*5）
        self.pool = nn.MaxPool2d(2, 2)        # 最大池化层，（卷积核尺寸，步长）
        self.conv2 = nn.Conv2d(6, 16, 5)      # 同样是卷积层
        self.conv3 = nn.Conv2d(16, 16, 5)      # 同样是卷积层
        self.fc1 = nn.Linear(16, 120) # 接着三个全连接层
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
 
    def forward(self, x):           # 这里定义前向传播的方法，为什么没有定义反向传播的方法呢？这其实就涉及到torch.autograd模块了，
        x = self.pool(F.relu(self.conv1(x)))  # F是torch.nn.functional的别名，这里调用了relu函数 F.relu()
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 16 )  # .view( )是一个tensor的方法，使得tensor改变形状。
                                    #  你会发现第一个全连接层的首参数是16*5*5，所以要保证能够相乘，在矩阵乘法之前就要把x调到正确的size
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
 
# 类定义完之后实例化net
net_0 = Net_0()
#%% 定义损失函数和优化器
import torch.optim as optim          #导入torch.potim模块
 
optimizer_0 = optim.SGD(net_0.parameters(), lr=0.002, momentum=0.9, weight_decay=0.0001)   #optim模块中的SGD梯度优化方式---随机梯度下降
loss_count_0=[]
ac_count_0 = []
#%% 训练
for epoch in range(20):  # 指定训练一共要循环几个epoch
    running_loss = 0.0  #定义一个变量方便我们对loss进行输出
    for i, data in enumerate(trainloader, 0): # enumerate是python的内置函数，既获得索引也获得数据
        inputs, labels = data   # data是从enumerate返回的data，包含数据和标签信息，分别赋值给inputs和labels
        inputs, labels = Variable(inputs), Variable(labels) # 将数据转换成Variable，第二步里面我们已经引入这个模块
                                                           
        optimizer_0.zero_grad()                # 要把梯度重新归零，因为反向传播过程中梯度会累加上一次循环的梯度   
        outputs = net_0(inputs)                # 把数据输进网络net
        loss = criterion(outputs, labels)    # 计算损失值
        loss.backward()                      # loss进行反向传播
        optimizer_0.step()                     # 当执行反向传播之后，把优化器的参数进行更新，以便进行下一轮
        # print statistics                   
        running_loss += loss.item()         # 从下面一行代码可以看出它是每循环0-1999共两千次才打印一次
        if i % 2000 == 1999:    
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))  # 然后再除以2000，就得到这两千次的平均损失值
            loss_count_0.append(running_loss/2000)
            running_loss = 0.0               # 这一个2000次结束后，就把running_loss归零，下一个2000次继续使用
            
            correct = 0   # 定义预测正确的图片数，初始化为0
            total = 0     # 总共参与测试的图片数，也初始化为0
            for data in testloader:  # 循环每一个batch
                images, labels = data
                outputs = net_0(Variable(images))  # 输入网络进行测试
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)          # 更新测试图片的数量
                correct += (predicted == labels).sum() # 更新正确分类的图片的数量
            ac = 100 * correct / total
            print('Accuracy of the network on the 10000 test images: %d %%' % (ac))     # 最后打印结果
            ac_count_0.append(ac)
            
print('Finished Training')
#%% 测试
plt.plot(loss_count, label='Original Training Loss')
plt.plot(loss_count_0, label='Improvement Training Loss')
plt.title('Loss')
plt.legend()
plt.show()

correct = 0   # 定义预测正确的图片数，初始化为0
total = 0     # 总共参与测试的图片数，也初始化为0
for data in testloader:  # 循环每一个batch
    images, labels = data
    outputs = net_0(Variable(images))  # 输入网络进行测试
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)          # 更新测试图片的数量
    correct += (predicted == labels).sum() # 更新正确分类的图片的数量
 
print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))          # 最后打印结果
 
plt.plot(ac_count, label='Original test accuracy')
plt.plot(ac_count_0, label='Improvement test accuracy')
plt.title('Accuracy')
plt.legend()
plt.show()








