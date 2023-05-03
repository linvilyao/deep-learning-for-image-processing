import os
import sys
import json
import time

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

from model import AlexNet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224), #随机裁剪到224*224
                                 transforms.RandomHorizontalFlip(), #水平方向随机翻转的函数
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
# getcwd获取当前文件所在目录，..返回上层目录
data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))  # get data root path
image_path = data_root + "/data_set/flower_data/"
assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
train_dataset = datasets.ImageFolder(root=image_path + "/train",
                                     transform=data_transform["train"])
train_num = len(train_dataset)
# 雏菊         蒲公英          玫瑰      向日葵         郁金香
# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
# 获取分类的名称所对应的索引
flower_list = train_dataset.class_to_idx
# 遍历flower_list字典，将key和val反过来，是为了预测之后返回的索引能直接使用字典对应到所属的类别
cla_dict = dict((val, key) for key, val in flower_list.items())
# 通过json将cla_dict字典编码成json的格式
json_str = json.dumps(cla_dict, indent=4)
# 打开class_indices.json文件，将json_str保存进去，为了方便预测时读取信息
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)

validate_dataset = datasets.ImageFolder(root=image_path + "/val",
                                        transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=4, shuffle=True,
                                              num_workers=0)

test_data_iter = iter(validate_loader)
test_image, test_label = test_data_iter.__next__()

# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
# imshow(utils.make_grid(test_image))

net = AlexNet(num_classes=5, init_weights=True)

net.to(device)
# CrossEntropyLoss针对多类别的损失交叉熵函数
loss_function = nn.CrossEntropyLoss()
# 调试用来查看模型的参数
# pata = list(net.parameters())
optimizer = optim.Adam(net.parameters(), lr=0.0002)

save_path = './AlexNet.pth'
# 历史最优准确率初始化
best_acc = 0.0
for epoch in range(10):
    # train
    # 使用到Dropout，想要实现的是在训练过程中失活一部分神经元，而不想在预测过程中失活
    # 因此通过net.train()和net.eval()来管理Dropout方法，在net.train()中开启Dropout方法，而在net.eval()中会关闭掉
    net.train()
    running_loss = 0.0
    t1 = time.perf_counter()
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # 打印在训练过程中的训练进度，len(train_loader)获取训练一轮所需要的步数step + 1获取当前的轮数
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "*" * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end=" ")
    print()
    print(time.perf_counter() - t1)


    # validate
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        for data_test in validate_loader:
            test_image, test_label = data_test
            outputs = net(test_image.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, test_label.to(device)).sum().item()
        accurate_test = acc / val_num
        if accurate_test > best_acc:
            best_acc = accurate_test
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, accurate_test))
print('Finished Training')
