
import torch.nn as nn
import torch
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self, num_class=10, init_weights=False):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4),     # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(3, 2),             # kernel_size, stride
        )

        self.conv2 = nn.Sequential(
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )

        self.conv3 = nn.Sequential(
            # 连续 3 个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.features = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
        )

        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_class),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, img):
        features = self.features(img)
        features = features.view(img.shape[0], -1)
        outputs = self.fc(features)
        return outputs

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                # 使用正态分布对输入张量进行赋值
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Linear):
                # 使 layer.weight 值服从正态分布 N(mean, std)，默认值为 0，1。通常设置较小的值。
                nn.init.normal_(layer.weight, 0, 0.01)
                # 使 layer.bias 值为常数 val
                nn.init.constant_(layer.bias, 0)


# # model 测试
# images = torch.rand([64, 1, 224, 224])      # 定义 shape
# model = AlexNet()                           # 实例化
# outputs = model(images)                     # 输入网络中
# print(outputs.shape)
# print(outputs)
