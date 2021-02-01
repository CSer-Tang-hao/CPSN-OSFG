import time
import torch
# tm = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
# epoch=1
# lr=0.01
# print('[{}] Accuracy: {:.2%}, std: {:.2%}'.format(tm,epoch, lr))
# print(torch.ones(1)*0.5)
# M=torch.randn(1,1,1,4,5)
# M_s2 = M.transpose(3,4)
# print(M_s2.size())
import torch.nn as nn
import torch.nn.functional as F
# global_max_pooling = nn.AdaptiveMaxPool2d(1)
# def similarity( support_x, query_x):
#     "Note that the size of support_x and query_x must be [n*b,n*s,64,19,19]"
#     # print(self.width,self.new_width)
#     # print(support_x.shape,query_x.shape)#torch.Size([4, 30, 5, 512, 6, 6]) torch.Size([4, 30, 5, 512, 6, 6])
#     count = 0
#     b, q, s, c, h, w = support_x.shape
#     for i in range(h-2):
#         for j in range(w-2):
#             column = support_x[:, :, :, :, i:i+3, j:j+3].view(-1,c,3,3) # [50,5,640]
#             columns = F.interpolate(column, (h, w), mode='bilinear' ,align_corners=True).view(b,q,s,c,h,w)
#
#             similarity = global_max_pooling(
#                 torch.cosine_similarity(query_x, columns, dim=-3).view(-1, s, h, w)).squeeze().view(b, q, s)  # [50,5]
#
#             similarity = similarity.unsqueeze(-1)  # [n*b,n*s,1]
#             if count == 0:
#                 out = similarity
#             else:
#                 out = torch.cat([out, similarity], dim=-1)
#             count = count + 1
#
#     "out -> [n*b,n*s,19*19]"
#
#     return out
#
# support=torch.randn(600,1024,6,6)
# # query=torch.randn(4,30,5,512,6,6)
# # out=similarity(support,query)
# # print(out.size())
#
# l1=nn.Conv2d(1024, 512, stride=1, kernel_size=3, bias=False)
# o=l1(support)
# print(o.shape)
import math
# a=torch.tensor([26.5544, 26.5922, 26.7418, 27.6719, 26.8466, 26.5544, 26.5544, 28.2380,
#         39.1059, 42.2250, 42.3170, 26.5544, 26.5544, 41.4818, 49.2508, 48.4032,
#         48.3838, 27.2651, 26.5544, 36.6901, 39.6650, 36.0645, 35.7298, 27.2956,
#         26.5544, 28.4556, 31.5151, 32.8130, 30.2144, 26.5544, 26.5544, 26.5544,
#         26.5968, 26.6775, 26.5544, 26.5544])
# print(a)
#
# a1=F.softmax(a, dim=-1)
# a2=F.softmax(3*a,dim=-1)
# print(a1,a2)
# a=torch.tensor([[1,2,3],[4,5,6]])
# print(a)
# index = torch.tensor([1,2])
# print(index[1])
#
# aa=torch.zeros(1,2)
# print(aa.shape)
# for i in range(2):
#
#         aa[:,i]=a[i,int(index[i])]
#
# print(aa)

weight=torch.randn(1500,36,36)
index = torch.ones(1500,36).long()
# print(index[0,:].numpy())
# exit(0)
select=torch.zeros(1500,36)
for i in range(1500):
        for j in range(36):
                select[i,j]=weight[i,j,index[i,j]]
                # print(select.shape)
                # print(select)

print(select.shape)
