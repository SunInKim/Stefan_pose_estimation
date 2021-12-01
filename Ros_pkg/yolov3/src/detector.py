#!/usr/bin/env python

from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import cv2

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torchvision.transforms as transforms

# ROS imports
import rospy
import std_msgs.msg
from rospkg import RosPack
from std_msgs.msg import UInt8
from sensor_msgs.msg import Image as ImageMsg
from geometry_msgs.msg import Polygon, Point32
from yolov3.msg import BoundingBox, BoundingBoxes
from cv_bridge import CvBridge, CvBridgeError

package = RosPack()
package_path = package.get_path('yolov3')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from torchvision import models
from math import pi
from PIL import Image


def draw_line(image, theta,x_p1,x_p3,y_p1,y_p3):

	width = x_p3-x_p1
	height = y_p3-y_p1
	if width > height:
		point = 0.25 * width
	else:
		point = 0.25 * height

	theta = theta*math.pi/180
	x1 = (x_p1+x_p3)/2
	y1 = (y_p1+y_p3)/2
	x2 = x1 + point*math.cos(theta)
	y2 = y1 -  point*math.sin(theta)

	img = cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (100,0,0), 3)
	img = cv2.line(img, (int(x1), int(y1)), (int(x1+point), int(y1)), (0,100,0), 3)

	return img


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
		# x = self.drop(x)	
		x = F.leaky_relu(self.fc2(x),negative_slope=0.2)
		# x = self.drop(x)	
		x = F.leaky_relu(self.fc3(x),negative_slope=0.2)
		
		sin_o, cos_o = torch.chunk(self.hard_tanh(self.fc4(x)), 2, dim=-1)

		return sin_o, cos_o
	

obj_list = ['back','bottom_black','bottom_white','frame_black','frame_white','side_black','side_white']
color_list = [(100,0,0),(0,100,0),(0,0,100),(100,100,0),(100,0,100),(0,100,100),(100,50,150)]

class Yolov3():
	def __init__(self):
		self.weights_path = "/home/irl-assembly/vision_ws/src/weights/pose_weights/yolov3.pth"
		rospy.loginfo("Found weights, loading %s", self.weights_path)

		# Raise error if it cannot find the model
		if not os.path.isfile(self.weights_path):
			raise IOError(('{:s} not found.').format(self.weights_path))

		# Load image parameter and confidence threshold
		self.image_topic_A = rospy.get_param('~image_topic_A', '/kinectA/rgb/image_raw')
		self.image_topic_B = rospy.get_param('~image_topic_B', '/kinectB/rgb/image_raw')
		self.confidence_th = rospy.get_param('~confidence', 0.85)
		self.nms_th = rospy.get_param('~nms_th', 0.3)

		# Load publisher topics
		self.detected_objects_topic_A = rospy.get_param('~detected_objects_topic_A')
		self.published_image_topic_A = rospy.get_param('~detections_image_topic_A')

		self.detected_objects_topic_B = rospy.get_param('~detected_objects_topic_B')
		self.published_image_topic_B = rospy.get_param('~detections_image_topic_B')

		# Load other parameters
		config_name = rospy.get_param('~config_name', 'yolov3-custom.cfg')
		self.config_path = os.path.join(package_path, 'config', config_name)
		classes_name = rospy.get_param('~classes_name', 'classes.names')
		self.classes_path = os.path.join(package_path, 'classes', classes_name)
		self.gpu_id = rospy.get_param('~gpu_id', 0)
		self.network_img_size = rospy.get_param('~img_size', 416)
		self.publish_image = rospy.get_param('~publish_image')

		self.camera_setting()

		# Initialize width and height
		self.h = 0
		self.w = 0

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
		#Set up model
		self.model = Darknet(self.config_path, img_size=self.network_img_size).to(self.device)

		self.model.load_state_dict(torch.load(self.weights_path))
		self.model.eval()# Set in evaluation mode

		# Load CvBridge
		self.bridge = CvBridge()

		self.classes = load_classes(self.classes_path)  # Extracts class labels from file
		self.classes_colors = {}

		self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

		self.vgg = models.vgg16(pretrained=True)
		self.vgg = self.vgg.cuda()
		self.features = self.vgg.features

		self.net_path = "/home/irl-assembly/vision_ws/src/weights/pose_weights/asdf.pth"
		
		self.ang_net = Angle_net()
		self.ang_net.cuda()
		self.ang_net.load_state_dict(torch.load(self.net_path, map_location="cuda:0"))

		self.transform = transforms.Compose(
			[transforms.ToTensor(),
			transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

		self.depth_sub_A = rospy.Subscriber("/kinectA/depth_to_rgb/image_raw", ImageMsg, self.callback_depth_A)
		self.depth_sub_B =rospy.Subscriber("/kinectB/depth_to_rgb/image_raw", ImageMsg, self.callback_depth_B)

		 # Define subscribers
		self.image_sub_A = rospy.Subscriber(self.image_topic_A, ImageMsg, self.imageCb_A, queue_size = 1, buff_size = 2**24)
		self.image_sub_B = rospy.Subscriber(self.image_topic_B, ImageMsg, self.imageCb_B, queue_size = 1, buff_size = 2**24)

		# Define publishers
		self.pub_A = rospy.Publisher(self.detected_objects_topic_A, BoundingBoxes, queue_size=10)
		self.pub_viz_A = rospy.Publisher(self.published_image_topic_A, ImageMsg, queue_size=10)

		self.pub_B = rospy.Publisher(self.detected_objects_topic_B, BoundingBoxes, queue_size=10)
		self.pub_viz_B = rospy.Publisher(self.published_image_topic_B, ImageMsg, queue_size=10)
		rospy.loginfo("Launched node for object detection")

		# Spin
		rospy.spin()


	def image_process(self, image):
		img = Image.fromarray(image)
		img = transforms.ToTensor()(img)
		img, _ = pad_to_square(img, 0)
		img = resize(img, self.network_img_size)
		img = img.unsqueeze(0)
		input_img = Variable(img.type(self.Tensor))

		return input_img

	def callback_depth_A(self, data):
		try:
			self.depth_raw_A = self.bridge.imgmsg_to_cv2(data,'passthrough')
		except CvBridgeError as e:
			print(e)

	def callback_depth_B(self, data):
		try:
			self.depth_raw_B = self.bridge.imgmsg_to_cv2(data,'passthrough')
		except CvBridgeError as e:
			print(e)

	def imageCb_A(self, data):
		  
			# Convert the image to OpenCV
			try:
				cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
			except CvBridgeError as e:
				print(e)
			# Initialize detection results
			detection_results = BoundingBoxes()
			detection_results.header = data.header
			detection_results.image_header = data.header

			image = self.image_process(cv_image)

			img_detections = []  # Stores detections for each image index

			with torch.no_grad():
				detections = self.model(image)
				detections = non_max_suppression(detections, self.confidence_th, self.nms_th)

			if detections[0] is not None:
				# Rescale boxes to original image
				detections = rescale_boxes(detections[0], self.network_img_size, cv_image.shape[:2])
				for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

					box_w = x2 - x1
					box_h = y2 - y1
					obj_ind = np.zeros(7)
					obj_ind[int(cls_pred.data.cpu())] = 1
					obj_ind = torch.FloatTensor(obj_ind)
					obj_ind = Variable(obj_ind.cuda())
					obj_name = obj_list[int(cls_pred.data)]

					obj = cls_pred.data.cpu()+1

					x_meter, y_meter = self.pixel_to_tool((x2+x1)/2.0,(y2+y1)/2.0,"A")

					x1, y1 = int(x1.data), int(y1.data)
					x2, y2 = int(x2.data), int(y2.data)					
					x1, y1 = max(0, x1), max(0, y1)
					
					img = np.asarray(cv_image)

					temp_img = img[y1:y2,x1:x2]

					cols = temp_img.shape[1]
					rows = temp_img.shape[0]
					if rows > cols:
						temp_img = cv2.resize(temp_img, (int(round(224*(float(cols)/float(rows)))), 224))
					else:
						temp_img = cv2.resize(temp_img, (224, int(round(224*(float(rows)/float(cols))))))

					new_cols = temp_img.shape[1]
					new_rows = temp_img.shape[0]
					temp_img = cv2.copyMakeBorder(temp_img,  int((224-new_rows)/2), int((224-new_rows)/2), int((224-new_cols)/2), int((224-new_cols)/2), cv2.BORDER_CONSTANT,value=(255,255,255))
		   
					temp_img = cv2.resize(temp_img,(224,224),interpolation=cv2.INTER_CUBIC)
					temp_img = Image.fromarray(temp_img)
					temp_img = self.transform(temp_img)
					temp_img = Variable(temp_img.cuda())

					sin_o, cos_o = self.ang_net(temp_img.unsqueeze(0),obj_ind.unsqueeze(0))
					sin_o = torch.transpose(sin_o,0,1)
					cos_o = torch.transpose(cos_o,0,1)

					output = torch.atan2(sin_o,cos_o)[0][0]/pi

					if obj == 4 or obj ==5:
						ang = (float(output.data) * 180.0/2.0)+90
					else:
						ang = float(output.data) * 180.0
					
					# Populate darknet message
					detection_msg = BoundingBox()
					detection_msg.xmin = x1
					detection_msg.xmax = x2
					detection_msg.ymin = y1
					detection_msg.ymax = y2
					detection_msg.angle = ang
					detection_msg.probability = conf
					detection_msg.x_meter = x_meter
					detection_msg.y_meter = y_meter
					x_dist= x2-x1
					y_dist= y2-y1
					dist = math.sqrt(x_dist**2 + y_dist**2)

					if(int(cls_pred) == 3):
						if(dist > 300):
							detection_msg.Class = "long_black"
						else:
							detection_msg.Class = "short_black"
					elif(int(cls_pred) == 4):
						if(dist > 300):
							detection_msg.Class = "long_white"
						else:
							detection_msg.Class = "short_white"

					else:
						detection_msg.Class = self.classes[int(cls_pred)]
					print(detection_msg)

					# Append in overall detection message
					detection_results.bounding_boxes.append(detection_msg)
			 
			# Publish detection results
			self.pub_A.publish(detection_results)

			# Visualize detection results
			if (self.publish_image):
				self.visualizeAndPublish_A(detection_results, cv_image)

			return True

	def imageCb_B(self, data):
		  
			# Convert the image to OpenCV
			try:
				cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
			except CvBridgeError as e:
				print(e)
			# Initialize detection results
			detection_results = BoundingBoxes()
			detection_results.header = data.header
			detection_results.image_header = data.header



			image = self.image_process(cv_image)
			img_detections = []  # Stores detections for each image index

			with torch.no_grad():
				detections = self.model(image)
				detections = non_max_suppression(detections, self.confidence_th, self.nms_th)

			if detections[0] is not None:
				# Rescale boxes to original image
				detections = rescale_boxes(detections[0], self.network_img_size, cv_image.shape[:2])
				for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

					box_w = x2 - x1
					box_h = y2 - y1
					obj_ind = np.zeros(7)
					obj_ind[int(cls_pred.data.cpu())] = 1
					obj_ind = torch.FloatTensor(obj_ind)
					obj_ind = Variable(obj_ind.cuda())

					obj_name = obj_list[int(cls_pred.data)]

					obj = cls_pred.data.cpu()+1

					x_meter, y_meter = self.pixel_to_tool((x2+x1)/2.0,(y2+y1)/2.0,"B")

					x1, y1 = int(x1.data), int(y1.data)
					x2, y2 = int(x2.data), int(y2.data)					
					x1, y1 = max(0, x1), max(0, y1)

					img = np.asarray(cv_image)					
					
					temp_img = img[y1:y2,x1:x2]

					cols = temp_img.shape[1]
					rows = temp_img.shape[0]
					if rows > cols:
						temp_img = cv2.resize(temp_img, (int(round(224*(float(cols)/float(rows)))), 224))
					else:
						temp_img = cv2.resize(temp_img, (224, int(round(224*(float(rows)/float(cols))))))

					new_cols = temp_img.shape[1]
					new_rows = temp_img.shape[0]
					temp_img = cv2.copyMakeBorder(temp_img,  int((224-new_rows)/2), int((224-new_rows)/2), int((224-new_cols)/2), int((224-new_cols)/2), cv2.BORDER_CONSTANT,value=(255,255,255))
		   
					temp_img = cv2.resize(temp_img,(224,224),interpolation=cv2.INTER_CUBIC)
					temp_img = Image.fromarray(temp_img)
					temp_img = self.transform(temp_img)
					temp_img = Variable(temp_img.cuda())

					sin_o, cos_o = self.ang_net(temp_img.unsqueeze(0),obj_ind.unsqueeze(0))
					sin_o = torch.transpose(sin_o,0,1)
					cos_o = torch.transpose(cos_o,0,1)

					output = torch.atan2(sin_o,cos_o)[0][0]/pi

					if obj == 4 or obj ==5:
						ang = (float(output.data) * 180.0/2.0)+90
					else:
						ang = float(output.data) * 180.0
					
					# Populate darknet message
					detection_msg = BoundingBox()
					detection_msg.xmin = x1
					detection_msg.xmax = x2
					detection_msg.ymin = y1
					detection_msg.ymax = y2
					detection_msg.angle = ang
					detection_msg.probability = conf
					detection_msg.x_meter = x_meter
					detection_msg.y_meter = y_meter
					x_dist= x2-x1
					y_dist= y2-y1
					dist = math.sqrt(x_dist**2 + y_dist**2)

					if(int(cls_pred) == 3):
						if(dist > 295):
							detection_msg.Class = "long_black"
						else:
							detection_msg.Class = "short_black"
					elif(int(cls_pred) == 4):
						if(dist > 295):
							detection_msg.Class = "long_white"
						else:
							detection_msg.Class = "short_white"

					else:
						detection_msg.Class = self.classes[int(cls_pred)]
					print(detection_msg)

					# Append in overall detection message
					detection_results.bounding_boxes.append(detection_msg)
			 
			# Publish detection results
			self.pub_B.publish(detection_results)

			# Visualize detection results
			if (self.publish_image):
				self.visualizeAndPublish_B(detection_results, cv_image)

			return True

	def visualizeAndPublish_A(self, output, imgIn):
		# Copy image and visualize
		imgOut = imgIn.copy()
		font = cv2.FONT_HERSHEY_SIMPLEX
		fontScale = 0.6
		thickness = 2
		for index in range(len(output.bounding_boxes)):
			label = output.bounding_boxes[index].Class
			x_p1 = output.bounding_boxes[index].xmin
			y_p1 = output.bounding_boxes[index].ymin
			x_p3 = output.bounding_boxes[index].xmax
			y_p3 = output.bounding_boxes[index].ymax
			angle = output.bounding_boxes[index].angle
			confidence = output.bounding_boxes[index].probability
			w = int(x_p3 - x_p1)
			h = int(y_p3 - y_p1)

			# Find class color
			if label in self.classes_colors.keys():
				color = self.classes_colors[label]
			else:
				# Generate a new color if first time seen this label
				color = np.random.randint(0,188,3)
				self.classes_colors[label] = color

			# Create rectangle
			cv2.rectangle(imgOut, (int(x_p1), int(y_p1)), (int(x_p3), int(y_p3)), (color[0],color[1],color[2]),2)
			imgOut = draw_line(imgOut,angle,x_p1,x_p3,y_p1,y_p3)
			text = ('{:s}: {:.3f}').format(label,confidence)
			cv2.putText(imgOut, text, (int(x_p1), int(y_p1-10)), font, fontScale, (0,0,0), thickness ,cv2.LINE_AA)

			# Create center
			center = (int(((x_p1)+(x_p3))/2),int(((y_p1)+(y_p3))/2))
			cv2.circle(imgOut,(center), 1, color, 3)

		# Publish visualization image
		image_msg = self.bridge.cv2_to_imgmsg(imgOut, "rgb8")
		self.pub_viz_A.publish(image_msg)

	def visualizeAndPublish_B(self, output, imgIn):
		# Copy image and visualize
		imgOut = imgIn.copy()
		font = cv2.FONT_HERSHEY_SIMPLEX
		fontScale = 0.6
		thickness = 2
		for index in range(len(output.bounding_boxes)):
			label = output.bounding_boxes[index].Class
			x_p1 = output.bounding_boxes[index].xmin
			y_p1 = output.bounding_boxes[index].ymin
			x_p3 = output.bounding_boxes[index].xmax
			y_p3 = output.bounding_boxes[index].ymax
			angle = output.bounding_boxes[index].angle
			confidence = output.bounding_boxes[index].probability
			w = int(x_p3 - x_p1)
			h = int(y_p3 - y_p1)

			# Find class color
			if label in self.classes_colors.keys():
				color = self.classes_colors[label]
			else:
				# Generate a new color if first time seen this label
				color = np.random.randint(0,188,3)
				self.classes_colors[label] = color

			# Create rectangle
			cv2.rectangle(imgOut, (int(x_p1), int(y_p1)), (int(x_p3), int(y_p3)), (color[0],color[1],color[2]),2)
			imgOut = draw_line(imgOut,angle,x_p1,x_p3,y_p1,y_p3)
			text = ('{:s}: {:.3f}').format(label,confidence)
			cv2.putText(imgOut, text, (int(x_p1), int(y_p1-10)), font, fontScale, (0,0,0), thickness ,cv2.LINE_AA)

			# Create center
			center = (int(((x_p1)+(x_p3))/2),int(((y_p1)+(y_p3))/2))
			cv2.circle(imgOut,(center), 1, color, 3)

		# Publish visualization image
		image_msg = self.bridge.cv2_to_imgmsg(imgOut, "rgb8")
		self.pub_viz_B.publish(image_msg)

	def camera_setting(self):

		self.z_cam_to_tool = 0
		self.cam2tool_x_offset = 0      
		self.cam2tool_y_offset = 0
		
		#  intrinsic parameter
		self.fx_A = 611.8367309570312
		self.fy_A = 611.8024291992188
		self.cx_A = 638.2330322265625
		self.cy_A = 366.6072998046875

		self.fx_B = 613.9232788085938
		self.fy_B = 613.8340454101562
		self.cx_B = 639.4866943359375
		self.cy_B = 365.3985595703125


	def pixel_to_tool(self, xc, yc, camera):

		x_pix = int(xc.cpu().item())
		y_pix = int(yc.cpu().item())

		if camera =="A":
			zc = self.depth_raw_A[y_pix][x_pix]
			z_rgb = zc * 1000
			x_rgb = z_rgb*(x_pix-self.cx_A)/self.fx_A           
			y_rgb = z_rgb*(y_pix-self.cy_A)/self.fy_A

		else:
			zc = self.depth_raw_B[y_pix][x_pix]
			z_rgb = zc * 1000
			x_rgb = z_rgb*(x_pix-self.cx_B)/self.fx_B           
			y_rgb = z_rgb*(y_pix-self.cy_B)/self.fy_B         

		x = x_rgb + self.cam2tool_x_offset
		y = y_rgb + self.cam2tool_y_offset

		return x/1000, y/1000


if __name__=="__main__":
	# Initialize node
	rospy.init_node("detector_manager_node")

	# Define detector object
	yolov3 = Yolov3()
