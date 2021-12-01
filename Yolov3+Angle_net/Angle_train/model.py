
import torch.nn as nn
import torch.nn.functional as F
import torch

class Angle_net(nn.Module):

	def __init__(self):
		super(Angle_net, self).__init__()
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
		x = self.drop(x)	
		x = F.leaky_relu(self.fc2(x),negative_slope=0.2)
		x = self.drop(x)	
		x = F.leaky_relu(self.fc3(x),negative_slope=0.2)
		
		sin, cos = torch.chunk(self.hard_tanh(self.fc4(x)), 2, dim=-1)

		angle = torch.atan2(sin,cos)

		return angle

