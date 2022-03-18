from __future__ import absolute_import
from __future__ import division

import torch
import math
from torch import nn
from torch.nn import functional as F
from torchvision import utils as vutils



class CPSN(nn.Module):
    def __init__(self, in_channel=512, temp=64):
        super(CPSN, self).__init__()

        # self.width = width
        # self.new_width = width
        self.in_features = self.width * self.width
        self.in_channels = in_channel

        self.soft_max = nn.Softmax(dim=-1)
        self.meta_learner = nn.Sequential(
            nn.Conv2d(in_channel, temp, stride=1, kernel_size=1, bias=False),
            nn.BatchNorm2d(temp),
            nn.ReLU(),
            nn.Conv2d(temp, 1, stride=1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))



    def forward(self, f1, f2):
        b1, n1, c, h, w = f1.size()
        b2, n2, c, h, w = f2.size()

        f1_ori = f1 # [4, 5, 64, 5, 5]
        f2_ori = f2 # [4, 75, 64, 5, 5]

        # f1=f1.unsqueeze(2).repeat(1,1,n2,1,1,1).transpose(1, 2)
        # [4, 5, 64, 5, 5]-->[4, 5, 1, 64, 5, 5]-->[4, 5, 75, 64, 5, 5]-->[4, 75, 5, 64, 5, 5]
        # f2=f2.unsqueeze(1).repeat(1,n1,1,1,1,1).transpose(1, 2)
        # [4, 75, 64, 5, 5]-->[4, 1, 75, 64, 5, 5]-->[4, 5, 75, 64, 5, 5]-->[4, 75, 5, 64, 5, 5]

        f1 = f1.unsqueeze(1).repeat(1, n2, 1, 1, 1, 1)
        f2 = f2.unsqueeze(2).repeat(1, 1, n1, 1, 1, 1)


        similar12, similar21, query_index, support_index = self.PSM(f1,f2)
        s1, s2 = self.PWG(similar12, similar21, query_index, support_index, f1_ori, f2_ori)

        return s1, s2, f1, f2

    def PSM(self, support_x, query_x):
        "Note that the size of support_x and query_x must be [n*b,n*s,64,19,19]"
        # print(self.width,self.new_width)

        b, q, s, c, h, w = support_x.shape
        support_x = F.normalize(support_x, p=2, dim=-3, eps=1e-12)
        query_x = F.normalize(query_x, p=2, dim=-3, eps=1e-12)
        support_x = support_x.view(b, q, s, c, h * w)
        query_x = query_x.view(b, q, s, c, h * w).transpose(3, 4) # [b, q, s, h * w, c]

        out_qs = torch.matmul(query_x, support_x) # S12
        out_sq = out_qs.transpose(3, 4) # S21

        out_qs, query_index = torch.max(out_qs, dim=-2) # [4, 75, 5, 25], [4, 75, 5, 25]
        out_sq, support_index = torch.max(out_sq, dim=-2)

        return out_qs, out_sq, query_index, support_index

    def softmax(self, x):
        x = F.softmax(x.reshape(x.size(0), x.size(1), -1), 2)
        return x

    def PWG(self, s12, s21, query_index, support_index, support, query):

        b, s, c, h, w = support.size()
        b, q, c, h, w = query.size()

        # Todo: get weight先meta learner再升维
        support = support.contiguous().view([b * s, self.in_channels, h, w])
        a1 = self.meta_learner(support)  # [20, 1, 5, 5]
        a1 = a1.unsqueeze(1).repeat(1, q, 1, 1, 1) # [20, 75, 1, 5, 5]

        query = query.contiguous().view([b * q, self.in_channels, h, w])
        a2 = self.meta_learner(query)  # [300, 1, 5, 5]
        a2 = a2.unsqueeze(2).repeat(1, 1, s, 1, 1) # [300, 1, 5, 5, 5]

        # Todo: sort query_weight
        a2_flatten = a2.view(-1, self.in_features) # [1500,36]
        query_index_flatten = query_index.view(-1, self.in_features) # [1500,36]
        for index in range(a2_flatten.shape[0]):
            a2_flatten[index] = a2_flatten[index][query_index_flatten[index]]
        a2_sort = a2_flatten.view(b, q, s, self.in_features) # [4, 75, 5, 25]

        # Todo: sort support_weight
        a1_flatten = a1.view(-1, self.in_features)
        support_index_flatten = support_index.view(-1, self.in_features)
        for index in range(a1_flatten.shape[0]):
            a1_flatten[index] = a1_flatten[index][support_index_flatten[index]]
        a1_sort = a1_flatten.view(b, q, s, self.in_features)

        # Todo: flatten x1,x2,f1_weight,f2_weight,f1_weight_sort,f2_weight_sort
        # s12=s21=[4, 75, 5, 25]
        s12 = s12.view(-1, h*w)
        s21 = s21.view(-1, h*w)

        a1 = a1.view(-1, h*w)
        a2 = a2.view(-1, h*w)
        a1_sort = a1_sort.view(-1, h*w)
        a2_sort = a2_sort.view(-1, h*w)

        # Todo: weight1就是maxpool query那一维

        weight12 = a1 * a2_sort
        weight21 = a1_sort * a2

        # Todo: get similar*weight
        s1 = (s12 * weight12).view(-1, s, h * w).mean(-1)
        s2 = (s21 * weight21).view(-1, s, h * w).mean(-1)


        return s1, s2


    # def PWG(self, x1, x2, query_index, support_index, f1, f2):
    #
    #     b, s, c, h, w = f1.size()  # torch.Size([2, 30, 5, 1024, 6, 6])
    #     q = f2.size(1)
    #
    #     # Todo: get weight先meta learner再升维
    #     f1 = f1.contiguous().view([b * s, self.in_channels, self.width, self.width])
    #     f1_weight = self.meta_learner(f1)  # torch.Size([300, 1, 6, 6])
    #     f1_weight = f1_weight.unsqueeze(1).repeat(1, q, 1, 1, 1)
    #     # f1_weight = f1_weight.contiguous().view([b, q, s, self.in_features])
    #     f2 = f2.contiguous().view([b * q, self.in_channels, self.width, self.width])
    #     f2_weight = self.meta_learner(f2)  # torch.Size([300, 1, 6, 6])
    #     f2_weight = f2_weight.unsqueeze(2).repeat(1, 1, s, 1, 1)
    #
    #     # Todo: get weight先升维再metalearn而
    #     # f1 = f1.contiguous().view([b * s * q, self.in_channels, self.width, self.width])
    #     # f1_weight = self.meta_learner(f1)  # torch.Size([300, 1, 6, 6])
    #     #
    #     # f1_weight = f1_weight.contiguous().view([b, q, s, self.in_features])
    #     # f2 = f2.contiguous().view([b * s * q, self.in_channels, self.width, self.width])
    #     # f2_weight = self.meta_learner(f2)  # torch.Size([300, 1, 6, 6])
    #     # f2_weight = f2_weight.contiguous().view([b, q, s, self.in_features])
    #     # Todo: sort query_weight
    #     f2_weight_sort_flatten = f2_weight.view(-1)
    #     query_index_flatten = query_index.view(-1)
    #     f2_weight_sort_flatten = f2_weight_sort_flatten[query_index_flatten]
    #     f2_weight_sort = f2_weight_sort_flatten.view(b, q, s, self.in_features)
    #     # Todo: sort support_weight
    #     f1_weight_sort_flatten = f1_weight.view(-1)
    #     support_index_flatten = support_index.view(-1)
    #     f1_weight_sort_flatten = f1_weight_sort_flatten[support_index_flatten]
    #     f1_weight_sort = f1_weight_sort_flatten.view(b, q, s, self.in_features)
    #     # Todo: flatten x1,x2,f1_weight,f2_weight,f1_weight_sort,f2_weight_sort
    #     x1 = x1.view(-1, self.in_features)
    #     x2 = x2.view(-1, self.in_features)
    #     f1_weight = f1_weight.view(-1, self.in_features)
    #     f2_weight = f2_weight.view(-1, self.in_features)
    #     f2_weight_sort = f2_weight_sort.view(-1, self.in_features)
    #     f1_weight_sort = f1_weight_sort.view(-1, self.in_features)
    #
    #     # Todo: weight1就是maxpool query那一维
    #
    #     weight1 = f1_weight * f2_weight_sort
    #     weight2 = f1_weight_sort * f2_weight
    #
    #     # Todo visualization weight-----------------------------------------------------
    #     # w1 = weight1.view(30, 5, 6, 6)
    #     # w2 = weight2.view(30, 5, 6, 6)
    #     # s1 = x1.view(30, 5, 6, 6)
    #     # s2 = x2.view(30, 5, 6, 6)
    #     # k1 = support_index_flatten.view(30, 5, 6, 6)
    #     # k2 = query_index_flatten.view(30, 5, 6, 6)
    #     # if batch_idx==3:
    #     #     for i in range(30):
    #     #         for j in range(5):
    #     #             print('{}------------------------------------------------------'.format(i * 5 + j))
    #     #             print('w1 is {}'.format(w1[i, j, :, :]))
    #     #             print('w2 is {}'.format(w2[i, j, :, :]))
    #     #             print('k1 is {}'.format(k1[i, j, :, :]))
    #     #             print('k2 is {}'.format(k2[i, j, :, :]))
    #     #             print('s1 is {}'.format(s1[i, j, :, :]))
    #     #             print('s2 is {}'.format(s2[i, j, :, :]))
    #     # Todo: get similar*weight
    #     out1 = (x1 * weight1).view(-1, s, h * w).mean(-1)
    #     out2 = (x2 * weight2).view(-1, s, h * w).mean(-1)
    #
    #     return out1, out2