# coding:utf-8
import torch
import torch.nn as nn
from torchvision import models


class FCN32s(nn.Module):
    def __init__(self, num_classes=21):
        super(FCN32s, self).__init__()
        vgg = models.vgg16()
        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())
        for f in features:
            if 'MaxPool' in f.__class__.__name__:
                f.ceil_mode = True
            elif 'ReLU' in f.__class__.__name__:
                f.inplace = True  # inplace，False表示新创建一个对象对其修改，True表示直接对这个对象进行修改。
        self.pool5 = nn.Sequential(*features)
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        self.fc6.weight.data.copy_(classifier[0].weight.data.view(4096, 512, 7, 7))
        self.fc6.bias.data.copy_(classifier[0].bias.data)
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.fc7.weight.data.copy_(classifier[3].weight.data.view(4096, 4096, 1, 1))
        self.fc7.bias.data.copy_(classifier[3].bias.data)
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        self.sequence = nn.Sequential(
            self.fc6, nn.ReLU(inplace=True), nn.Dropout(),
            self.fc7, nn.ReLU(inplace=True), nn.Dropout(),
            self.score_fr
        )
        self.upscore = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32, bias=False)

    def forward(self, x):
        x = self.pool5(x)
        x = self.sequence(x)
        x = self.upscore(x)
        return x


class FCN16s(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass


class FCN8s(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass


net = FCN32s()
