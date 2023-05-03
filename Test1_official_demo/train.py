# noinspection PyUnresolvedReferences
import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
#50000张训练照片
transet = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=transform)

trainloader = torch.utils.data.DataLoader(transet, batch_size=36, shuffle=True, num_workers=0)


#10000张测试图片
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

test_data_iter = iter(testloader)
test_image, test_label = test_data_iter.__next__()

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'flog', 'horse', 'ship', 'truck')


def imshow(img):
    img = img / 2 + 0.5   #unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


#print labels
print(" ".join('%5s' % classes[test_label[j]] for j in range(4)))
#show images
imshow(torchvision.utils.make_grid(test_image))


net = LeNet()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
# 将训练集训练多少次，这里为5次
for epoch in range(5):
    # 用来累计在训练过程中的损失
    running_loss = 0.0
    # 遍历训练集样本
    # enumerate函数不仅能返回每一批的数据data，还能返回这一批data所对应的步数index，相当于C++中的枚举
    for step, data in enumerate(trainloader, start=0):
        #输入的图像及标签
        inputs, labels = data
        # 将历史损失梯度清零
        # 为什么每计算一个batch，就需要调用一次optimizer.zero._grad()
        # 如果不清除历史梯度，就会对计算的历史梯度进行累加（通过这个特性你能够变相实现一个很大batch数值的训练），主要还是硬件设备受限，防止爆内存
        optimizer.zero_grad()
        # 将我们得到的数的图片输入到网络进行正向传播，得到输出
        outputs = net(inputs)
        # 通过定义的loss_function来计算损失，outputs：网络预测的值，labels：真实标签
        loss = loss_function(outputs, labels)
        # 对loss进行反向传播
        loss.backward()
        # 通过优化器optimizer中step函数进行参数更新
        optimizer.step()

        # 打印的过程
        running_loss += loss.item()
        # 每隔500步，打印一次数据信息
        if step % 500 == 499:
            # with是一个上下文管理器，意思是在接下来的计算过程中，不要去计算每个节点的误差损失梯度
            # 否则会自动生成前向的传播图，会占用大量内存，测试时应该禁用
            with torch.no_grad():
                outputs = net(test_image)  # [batch, 10]
                # predict_y寻找outputs中数值最大的，也就是最有可能的标签类型
                # dim：第几个维度，第0隔维度时batch，第1隔维度指10隔标签结果；
                # [1]指只需要知道index即可，不需要知晓具体的值
                predict_y = torch.max(outputs, dim=1)[1]
                # 将预测的标签类别和真实的标签类别进行比较，相同的地方返回1，不相同返回0
                # 使用求和函数，得出在本次预测对了多少个样本
                # tensor得到的并不是数值，item()才可以拿到
                accuracy = torch.eq(predict_y, test_label).sum().item() / test_label.size(0)    # 数据准确率
                # 迭代到第几轮 在某一轮的多少步 训练过程中的累加误差 测试样本的准确率
                print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                        (epoch + 1, step + 1, running_loss / 500, accuracy))
                running_loss = 0.0

print('Finished Training')
# 对模型进行保存
save_path = './Lenet.pth'
torch.save(net.state_dict(), save_path)

