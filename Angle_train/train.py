import os
import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
from math import pi
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import models


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" 

class AngleTrainDataset(Dataset):

	def __init__(self, transform):
		self.transform = transform
		self.data_path = './data/train'
		self.image_path = os.path.join(self.data_path, 'Images')
		self.label_path = os.path.join(self.data_path, 'labels')
		self.image_list = os.listdir(self.image_path)
		
	def __getitem__(self, index):

		img = cv2.imread('%s/%s'%(self.image_path, self.image_list[index]))
		cols = img.shape[1]
		rows = img.shape[0]
		if img.shape[0] > img.shape[1]:
			img = cv2.resize(img,(int(round(224*(float(cols)/float(rows)))), 224))
		else:
			img = cv2.resize(img, (224, int(round(224*(float(rows)/float(cols))))))

		new_cols = img.shape[1]
		new_rows = img.shape[0]
		img = cv2.copyMakeBorder(img,  int((224-new_rows)/2), int((224-new_rows)/2),int((224-new_cols)/2), int((224-new_cols)/2), cv2.BORDER_CONSTANT,value=(0,0,0))
		img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
		img = Image.fromarray(img)
		transformed_img = self.transform(img)

		label = open('%s/%s'%(self.label_path, self.image_list[index].replace('png','txt').replace('jpg','txt')), 'r')
		line = label.readline()
		line = line.split(' ')
		angle = line[0]
		obj = line[1]
		angle = int(angle.strip())
		obj = int(obj.strip())
		obj_ind = np.zeros(7)
		obj_ind[obj-1] = 1
		
		if obj == 4 or obj ==5:
			angle = 2*(angle-90)/180.0

		else:
			angle /= 180.0

		obj_ind = torch.FloatTensor(obj_ind)

		return transformed_img, obj_ind, angle


	def __len__(self):		
		return len(self.image_list)


class AngleNet(nn.Module):

	def __init__(self):
		super(AngleNet, self).__init__()
		vgg = models.vgg16(pretrained=True)
		self.feature = vgg.features

		self.conv1 = nn.Conv2d(512, 1024, 7)
		
		self.fc1 = nn.Linear(1024+7, 512) 
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, 128)
		self.fc4 = nn.Linear(128, 2)

		self.hard_tanh = nn.Hardtanh(min_val=-1,max_val=1)

	def forward(self, x, a):

		x = self.feature(x)
		
		x = F.leaky_relu(self.conv1(x),negative_slope=0.2)
		x = x.view(-1,1024)
		x = torch.cat((x,a), dim=1)	

		x = F.leaky_relu(self.fc1(x),negative_slope=0.2)
		x = F.leaky_relu(self.fc2(x),negative_slope=0.2)
		x = F.leaky_relu(self.fc3(x),negative_slope=0.2)

		sin, cos = torch.chunk(self.hard_tanh(self.fc4(x)), 2, dim=-1)

		angle = torch.atan2(sin,cos)

		return angle

transform = transforms.Compose(
	[transforms.ToTensor(),
	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = AngleTrainDataset(transform)
train_loader = DataLoader(dataset=dataset, batch_size=2, shuffle=True, num_workers=0)

net = AngleNet()
net.cuda()

learning_rate = 2e-4

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)


for epoch in range(1000):

	for i, data in enumerate(train_loader, 0):
		inputs, obj, labels = data
		gt_obj = obj
		inputs, labels, obj = Variable(inputs.cuda()), Variable(labels.cuda()), Variable(obj.cuda())
		labels = labels.float().unsqueeze(0)
		optimizer.zero_grad()
		outputs = net(inputs,obj)
		outputs = torch.transpose(outputs,0,1)

		loss = criterion(outputs, pi*labels)
		
		loss.backward()
		optimizer.step()

		print("%i, loss:%f, label:%s, output:%s"%(i, float(loss.data), float(pi*labels[0][0]), float(outputs[0][0])))

	print("EPOCH : %d/1000"%(epoch+1))
	if ((epoch+1) % 50 == 0) :
		save_path = "./models/epoch_%.2i.pth"%epoch
		torch.save(net.state_dict(), save_path)
print('Finished Training')