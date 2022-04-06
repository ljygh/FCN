import time
import numpy as np
import torch
import logging

from torch import nn
from torch.utils.data import random_split, DataLoader
# import visdom
# vis = visdom.Visdom()

from dataset import CustomDataset
from model.fcn import FCNs

CUDA = torch.cuda.is_available()

def validate(**kwargs):
    # Retrieve training configuration
    data_loader = kwargs['data_loader']
    mymodel = kwargs['mymodel']
    criterion = kwargs['criterion']
    verbose = kwargs['verbose']

    #mymodel.eval()

    image, target = next(iter(data_loader))
    if CUDA:
        image = image.cuda()
        target = target.cuda()
    with torch.no_grad():
        output = mymodel(image)
        loss = criterion(output, target)
        print(loss)

    pred = output.data.cpu().numpy()
    pred = np.argmin(pred, axis=1)
    t = np.argmin(target.cpu().numpy(), axis=1)
    # print(pred[0])
    # print(t[0])
    # vis.close()
    # vis.images(pred[:, None, :, :], opts=dict(title='pred'))
    # vis.images(t[:, None, :, :], opts=dict(title='target'))


mymodel = FCNs(2)
checkpoint = torch.load('models/001.ckpt')
state_dict = checkpoint['state_dict']
mymodel.load_state_dict(state_dict)
if torch.cuda.is_available():
    mymodel.to(torch.device("cuda"))
    mymodel = nn.DataParallel(mymodel)

custom_dataset = CustomDataset()
train_size = int(0.9 * len(custom_dataset))
val_size = len(custom_dataset) - train_size
train_set, val_set = random_split(custom_dataset, [train_size, val_size])
val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

optimizer = torch.optim.Adam(mymodel.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

validate(data_loader=val_loader,
        mymodel=mymodel,
        criterion=criterion,
        verbose=100)