
import os
import sys
import cv2
import argparse
import numpy as np

from math import pi
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import models

from model import Angle_net
from dataset import AngleTrainDataset


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" 

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--save_interval", type=int, default=50, help="interval to save")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
opt = parser.parse_args()

transform = transforms.Compose(
	[transforms.ToTensor(),
	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = AngleTrainDataset(transform)
train_loader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)

net = Angle_net()
net.cuda()

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=opt.lr)


for epoch in range(opt.n_epochs):

	for i, data in enumerate(train_loader, 0):
		inputs, obj, labels = data
		gt_obj = obj
		inputs, labels, obj = inputs.cuda(), labels.cuda(),obj.cuda()
		optimizer.zero_grad()
		outputs = net(inputs,obj)

		loss = criterion(outputs, pi*labels)
		loss.backward()
		optimizer.step()

		print("%i, loss:%f, label:%s, output:%s"%(i, float(loss.data), float(pi*labels[0][0]), float(outputs[0][0])))

	print("EPOCH : %d/1000"%(epoch+1))
	if ((epoch+1) % opt.save_interval == 0) :
		save_path = "./models/epoch_%.2i.pth"%epoch
		torch.save(net.state_dict(), save_path)
print('Finished Training')