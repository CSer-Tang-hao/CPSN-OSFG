#%%

import torch
import numpy as np
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision import transforms
from matplotlib import pyplot as plt
from PIL import Image

import matplotlib.pyplot as plt


w1 = torch.tensor([[24.3661, 27.0947, 27.1134, 24.3661, 24.3661, 24.3661],
        [24.3661, 31.8008, 31.7137, 28.9405, 29.1490, 24.3661],
        [28.6857, 37.1992, 37.9529, 36.6463, 29.3109, 24.3661],
        [28.1230, 37.7275, 39.5223, 37.9417, 29.5916, 25.0633],
        [24.3661, 31.0254, 33.4551, 32.2618, 27.1874, 24.3771],
        [24.3661, 24.3661, 28.1230, 24.3661, 24.3661, 24.3661]])
w2 = torch.tensor([[24.3661, 24.3661, 24.3661, 24.3661, 24.3661, 24.3661],
        [24.3661, 26.0700, 27.5901, 28.0658, 27.0947, 24.3661],
        [24.3661, 28.6178, 30.3466, 36.9394, 30.2268, 24.4542],
        [24.3696, 29.8080, 36.8628, 37.4835, 36.4518, 24.3665],
        [24.3661, 28.1230, 31.0254, 30.8451, 30.6113, 24.3661],
        [24.3661, 24.3661, 24.3661, 24.3661, 24.3661, 24.3661]])
s1 = torch.tensor([[0.8897, 0.8714, 0.8583, 0.8994, 0.8497, 0.8759],
        [0.9313, 0.9439, 0.9485, 0.9405, 0.9090, 0.8880],
        [0.9458, 0.9422, 0.9442, 0.9406, 0.8896, 0.8644],
        [0.9276, 0.9323, 0.9441, 0.9136, 0.9081, 0.8285],
        [0.9088, 0.9079, 0.9111, 0.8737, 0.8351, 0.7615],
        [0.8714, 0.8866, 0.9050, 0.8913, 0.8508, 0.8738]])
s2 = torch.tensor([[0.8897, 0.8860, 0.8750, 0.8894, 0.8194, 0.8759],
        [0.9313, 0.9269, 0.9485, 0.9450, 0.9405, 0.8994],
        [0.9458, 0.9405, 0.9442, 0.9406, 0.9243, 0.9090],
        [0.9276, 0.9323, 0.9441, 0.9265, 0.9081, 0.8512],
        [0.9088, 0.9011, 0.9081, 0.9040, 0.8725, 0.8716],
        [0.8738, 0.8724, 0.8874, 0.9050, 0.8866, 0.8714]])
w1 = F.softmax(0.1*w1.view(-1),dim=0).view(6,6).numpy()
w2 = F.softmax(0.1*w2.view(-1),dim=0).view(6,6).numpy()
s1 = F.softmax(0.1*s1.view(-1),dim=0).view(6,6).numpy()
s2 = F.softmax(0.1*s2.view(-1),dim=0).view(6,6).numpy()
# print(type(w1),type(w2))
# exit(0)
# plt.matshow(w1, cmap=plt.get_cmap('Greens'), alpha=1)  # , alpha=0.3
# plt.show()
# # w3 = w1 * w2
# # plt.matshow(w3, cmap=plt.get_cmap('Greens'), alpha=20)  # , alpha=0.3
# # plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#fig=plt.figure(figsize=(10,8))
fig, ax1 = plt.subplots(figsize = (10, 8),nrows=1)
h=sns.heatmap(w1, annot=False,cmap="Greens",fmt='d',linewidths=1.5)
plt.show()

# cubehelix map颜色
# cmap = sns.cubehelix_palette(start = 1.5, rot = 3, gamma=0.8, as_cmap = True)
# sns.heatmap(w1, linewidths = 1.5, ax = ax1, cmap="Greens")
# ax1.set_title('w21')
# ax1.set_xlabel('')
# plt.show()