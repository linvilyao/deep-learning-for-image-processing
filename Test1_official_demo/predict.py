import torch
import torchvision.transforms as transforms
from PIL import Image

from model import LeNet


transform = transforms.Compose(
    [transforms.Resize((32, 32)),   #将图像缩放在32×32的大小
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 实例化LeNet
net = LeNet()
# 使用load_state_dict载入更改保存的Lenet.pth
net.load_state_dict(torch.load('Lenet.pth'))
# 载入之后，根据python的PIL import Image的，模块，取载入图像
im = Image.open('1.jpg')
# 预处理
im = transform(im)  # [C, H, W]
# Tensor规定时需要4个维度，但transform输出仅有3个，因此需要在index = 0处增加一个新的维度
im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

with torch.no_grad():
    # 将图像传入网络
    outputs = net(im)
    # 寻找输出中的最大值
    predict = torch.max(outputs, dim=1)[1].numpy()
# 将index索引传入到classes，得出类别
print(classes[int(predict)])

