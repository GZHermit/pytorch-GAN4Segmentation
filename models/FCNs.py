# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
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

        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)
        self.fc6.weight.data.copy_(classifier[0].weight.data.view(4096, 512, 7, 7))
        self.fc6.bias.data.copy_(classifier[0].bias.data)

        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.fc7.weight.data.copy_(classifier[3].weight.data.view(4096, 4096, 1, 1))
        self.fc7.bias.data.copy_(classifier[3].bias.data)

        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        self.score_fr.weight.data.zero_()
        self.score_fr.bias.data.zero_()

        self.sequence = nn.Sequential(
            self.fc6, nn.ReLU(inplace=True), nn.Dropout(),
            self.fc7, nn.ReLU(inplace=True), nn.Dropout(),
            self.score_fr
        )

        self.upscore = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32, padding=16, bias=False)

    def forward(self, x):
        # x_size = x.size()
        x = self.pool5(x)
        x = self.sequence(x)
        x = self.upscore(x)
        return x
        # return x[:, :, 19:(19 + x_size[2]), 19:(19 + x_size[2])].contiguous()


class FCN16s(nn.Module):
    def __init__(self, num_classes=21):
        super(FCN16s, self).__init__()
        vgg = models.vgg16()
        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())
        for f in features:
            if 'MaxPool' in f.__class__.__name__:
                f.ceil_mode = True
            elif 'ReLU' in f.__class__.__name__:
                f.inplace = True  # inplace，False表示新创建一个对象对其修改，True表示直接对这个对象进行修改。
        self.pool4 = nn.Sequential(*features[:24])
        self.pool5 = nn.Sequential(*features[24:])

        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool4.weight.data.zero_()
        self.score_pool4.bias.data.zero_()

        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)
        self.fc6.weight.data.copy_(classifier[0].weight.data.view(4096, 512, 7, 7))
        self.fc6.bias.data.copy_(classifier[0].bias.data)

        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.fc7.weight.data.copy_(classifier[3].weight.data.view(4096, 4096, 1, 1))
        self.fc7.bias.data.copy_(classifier[3].bias.data)

        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        self.score_fr.weight.data.zero_()
        self.score_fr.bias.data.zero_()

        self.sequence = nn.Sequential(
            self.fc6, nn.ReLU(inplace=True), nn.Dropout(),
            self.fc7, nn.ReLU(inplace=True), nn.Dropout(),
            self.score_fr
        )

        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.upscore16 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=32, stride=16, padding=8, bias=False)

    def forward(self, x):
        pool4 = self.pool4(x)
        pool5 = self.pool5(pool4)
        score_fr = self.sequence(pool5)
        upscore2 = self.upscore2(score_fr)
        score_pool4 = self.score_pool4(0.01 * pool4)
        upscore16 = self.upscore16(score_pool4 + upscore2)
        return upscore16


class FCN8s(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass


# net = FCN16s()
# inputs = Variable(torch.randn(1, 3, 224, 224))
# output = net(inputs)
# print(output.size())
