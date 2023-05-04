import torch
import torch.nn as nn

# 创建类AlexNet，继承于父类nn.Module
class AlexNet(nn.Module):
    # 通过初始化函数来定义AlexNet网络在正向传播过程中所需要使用到的一些层结构
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        # 这里与Pytorch官方demo不一样的是：使用到nn.Sequential模块
        # nn.Sequential能够将一系列的层结构进行打包，组合成一个新的结构，在这取名为features
        # features代表专门用于提取图像特征的结构
        # 为什么使用nn.Sequential模块？
        #   精简代码，减少工作量
        self.features = nn.Sequential(
            # 卷积核大小：11；卷积核个数原模型是96：由于数据集较小和加快运算速度，因此这里取一半48，经检测正确率相差不大；
            #   输入的图片是RGB的彩色图片：3；
            # padding有两种写的类型：一种是整型，一种是tuple类型。当padding=1时，代表在图片上下左右分别补一个单位的0.
            #   如果传入的是tuple(1, 2)：1代表上下方各补一行0；2表示左右两侧各补两列0.
            # 如果想要实现第一层的padding在上一堂课中讲到，是在最左边补一列0，最上面补一行0，最右边补两列0，最下面补两行0。
            #   nn.ZeroPad2d((1,2,1,2))：左侧补一列，右侧补两列，上方补一行，下方补两行
            # 这里使用padding = 2，按照公式计算出来结果为55.25，在Pytorch中如果计算结果为小数，会自动将小数点去掉
            # input[3, 224, 224]    output[48, 55, 55]
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),
            # inplace参数理解为Pytorch通过一种方法增加计算量，但降低内存使用
            nn.ReLU(inplace=True),
            # output[48, 27, 27]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # output[128, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # output[128, 13, 13]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # output[192, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # output[192, 13, 13]
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # output[128, 13, 13]
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # output[128, 6, 6]
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # classifier包含之后的三层全连接层
        self.classifier = nn.Sequential(
            # p代表失活比例，默认为0.5
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            # num_classes数据集类别的个数，5
            nn.Linear(2048, num_classes),
        )
        # 当搭建网络过程中传入初始化权重init_weights=trus，会进入到初始化权重函数
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)   #展平处理
        x = self.classifier(x)
        return x
    # 初始化权重函数
    def _initialize_weights(self):
        # 遍历modules模块，modules定义中：返回一个迭代器，迭代器中会遍历网络中所有的模块
        # 换而言之，通过self.modules()，会迭代定义的每一个层结构
        for m in self.modules():
            # 遍历层结构之后，判断属于哪一个类别
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                # 如果偏置不为0的话，就以0来作为初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                # 如果传进来的实例是全连接层，那么会通过normal_（正态分布）给权重weight赋值，均值=0，方差=0.01，偏置初始化0
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)