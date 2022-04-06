import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import os

np.set_printoptions(threshold=np.inf)
from torch.utils.data import DataLoader

from dataset import CustomDataset
from model import fcn


test_set = CustomDataset("data/testImages", mode="test")
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)
img, path = next(iter(test_loader))
mymodel = fcn.FCNs(2)
checkpoint = torch.load('models/005.ckpt')
state_dict = checkpoint['state_dict']
mymodel.load_state_dict(state_dict)
if torch.cuda.is_available():
    mymodel.to(torch.device("cuda"))
    mymodel = nn.DataParallel(mymodel)
mymodel.eval()
img = img.cuda()
with torch.no_grad():
    output = mymodel(img)

# output = output.data.cpu()
# output = torch.squeeze(output, 1)
# output = torch.sigmoid(output)
# pred = output.numpy()
# # pred.astype('uint8') * 255
# print(pred[0])
# mean = np.mean(pred)
# print(mean)
# for i in pred:
#     for j in i:
#         for k in range(len(j)):
#             if j[k] >= mean:
#                 j[k] = 1
#             else:
#                 j[k] = 0

# output = torch.sigmoid(output)
# output = output[:, 0, :, :]
# output = torch.squeeze(output, 1)
pred = output.data.cpu().numpy()
pred = np.argmin(pred, axis=1)

for j, p in enumerate(path):
    im = pred.astype('uint8')[j] * 255
    im = Image.fromarray(pred.astype('uint8')[j] * 255, "L")
    im = im.resize((320, 240))
    im.save(os.path.join("data/testPreds", p.split("\\")[-1]))

# pred = pred.astype('uint8')[0]*255
# im = Image.fromarray(pred, "L")
# im.save(r'D:\Ljy\Courses files\Ecust\Bachelor thesis\数据\Smoke\pred.jpg')
# plt.imshow(pred[0])
# plt.show()
