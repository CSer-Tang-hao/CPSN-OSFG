import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models.resnet12 import resnet12
from models.conv4 import ConvNet4
from models.CPSN import CPSN


# def one_hot(labels_train):
#     labels_train = labels_train.cpu()
#     nKnovel = 5
#     labels_train_1hot_size = list(labels_train.size()) + [nKnovel,]
#     labels_train_unsqueeze = labels_train.unsqueeze(dim=labels_train.dim())
#     labels_train_1hot = torch.zeros(labels_train_1hot_size).scatter_(len(labels_train_1hot_size) - 1, labels_train_unsqueeze, 1)
#     return labels_train_1hot
#
# def cos_dist(x, y):
#
#     m, n = x.size(0), y.size(0)
#
#     xx = F.normalize(x, p=2, dim=1, eps=1e-12).unsqueeze(1)
#     yy = F.normalize(y, p=2, dim=1, eps=1e-12).unsqueeze(0)
#
#     distmat = 7 * torch.sum(xx * yy, dim=2)
#
#     return distmat

class Model(nn.Module):
    def __init__(self, scale_cls, num_classes=64, backbone='C'):
        super(Model, self).__init__()
        self.scale_cls = scale_cls
        self.backbone = backbone
        if self.backbone == 'R':
            print('Using ResNet12')
            self.base = resnet12()
            # self.width = 6
            self.in_channel = 512
            self.temp = 64
        else:
            print('Using Conv64')
            self.base = ConvNet4()
            # self.width = 5
            self.in_channel = 64
            self.temp = 8

        self.cpsn = CPSN(in_channel=self.in_channel, temp=self.temp)

        self.nFeat = self.base.nFeat
        self.clasifier = nn.Conv2d(self.nFeat, num_classes, kernel_size=1) 

    # def test(self, ftrain, ftest, out1, out2):
    #     ftest = ftest.mean(4)
    #     ftest = ftest.mean(4)
    #     ftest = F.normalize(ftest, p=2, dim=ftest.dim()-1, eps=1e-12)
    #     ftrain = F.normalize(ftrain, p=2, dim=ftrain.dim()-1, eps=1e-12)
    #     scores = self.scale_cls * torch.sum(ftest * ftrain, dim=-1)
    #
    #     out = 0.5 * out1 + 0.5 * out2

        # return out

    def forward(self, xtrain, xtest, ytrain, ytest):
        # xtrain [4, 5, 3, 84, 84] ytrain [4, 5, 5]
        # xtest [4, 75, 3, 84, 84] ytest[4, 75, 5]

        batch_size, num_train = xtrain.size(0), xtrain.size(1)
        num_test = xtest.size(1)
        K = ytrain.size(2)
        ytrain = ytrain.transpose(1, 2)

        xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))# [20, 3, 84, 84]
        xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))# [300, 3, 84, 84]

        x_all = torch.cat((xtrain, xtest), 0)
        f= self.base(x_all) # [320, 64, 5, 5]

        ftrain = f[:batch_size * num_train]
        ftrain = ftrain.view(batch_size, num_train, -1) # [4, 5, 1600]

        # Getting Prototype
        ftrain = torch.bmm(ytrain, ftrain)
        ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain))
        ftrain = ftrain.view(batch_size, -1, *f.size()[1:])

        ftest = f[batch_size * num_train:]
        ftest = ftest.view(batch_size, num_test, *f.size()[1:])

        # CPSN:
        # input：
        # ftrain [4, 5, 64, 5, 5] ftest[4, 75, 64, 5, 5]
        out1, out2, ftrain, ftest = self.cpsn(ftrain, ftest)
        # [300, 5] [300, 5] [4, 75, 5, 64, 5, 5] [4, 75, 5, 64, 5, 5]

        # ftrain = ftrain.mean(4)
        # ftrain = ftrain.mean(4) # [4, 75, 5, 64]
        #
        if not self.training:
            return 0.5 * out1 + 0.5 * out2
        #
        # h,w=ftest.size(-2),ftest.size(-1)
        # ftest_norm = F.normalize(ftest, p=2, dim=3, eps=1e-12) # [4, 75, 5, 64, 5, 5]
        #
        # ftrain_norm = F.normalize(ftrain, p=2, dim=3, eps=1e-12)
        # ftrain_norm = ftrain_norm.unsqueeze(4)
        # ftrain_norm = ftrain_norm.unsqueeze(5) # [4, 75, 5, 64, 1, 1]
        #
        # cls_scores = self.scale_cls * torch.sum(ftest_norm * ftrain_norm, dim=3)
        # cls_scores = cls_scores.view(batch_size * num_test, *cls_scores.size()[2:])
        h, w = ftest.size(-2), ftest.size(-1)
        ftest = ftest.view(batch_size, num_test, K, -1)
        ftest = ftest.transpose(2, 3)
        ytest = ytest.unsqueeze(3)
        ftest = torch.matmul(ftest, ytest)
        ftest = ftest.view(batch_size * num_test, -1, h, w)

        ytest = self.clasifier(ftest)

        return ytest, out1, out2






