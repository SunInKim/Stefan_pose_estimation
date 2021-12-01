
import os
import sys
import cv2
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class AngleTrainDataset(Dataset):

	def __init__(self, transform):
		self.transform = transform
		self.data_path = './data/train'
		self.image_path = os.path.join(self.data_path, 'Images')
		self.label_path = os.path.join(self.data_path, 'labels')
		self.image_list = os.listdir(self.image_path)
		
	def __getitem__(self, index):

		#이미지 불러오고 padding 추가한 후 resize
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

		return transformed_img, obj_ind, [angle]


	def __len__(self):		
		return len(self.image_list)
