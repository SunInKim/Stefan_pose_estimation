import numpy as np
import os
import math
from random import *
# import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import parmap
import imutils
import cv2
from multiprocessing import Manager

from aug import *

obj_name = ['back','bottom_black','bottom_white','frame_balck','frame_white','side_black','side_white']
yolo_save_folder = './Yolo_train/data/custom/train/'
ang_save_folder = './Angle_train/data/train/'
image_path = './stephan_data/data/yolo_mask/'
bg_path = './stephan_data/bg/'

num_data = 3000

if yolo_save_folder is not None:
	YOLO_SAVE = True
else:
	YOLO_SAVE = False

if ang_save_folder is not None:
	ANG_SAVE = True
else:
	ANG_SAVE = False

if YOLO_SAVE:
	if not os.path.exists(yolo_save_folder + 'images'):
		os.mkdir(yolo_save_folder + 'images')
	yolo_image_save = os.path.join(yolo_save_folder,'images')

	if not os.path.exists(yolo_save_folder + 'labels'):
		os.mkdir(yolo_save_folder + 'labels')
	yolo_label_save = os.path.join(yolo_save_folder,'labels')

if ANG_SAVE:
	if not os.path.exists(ang_save_folder + 'labels'):
		os.mkdir(ang_save_folder + 'labels')
	ang_label_save = os.path.join(ang_save_folder,'labels')

	if not os.path.exists(ang_save_folder + 'Images'):
		os.mkdir(ang_save_folder + 'Images')
	ang_image_save = os.path.join(ang_save_folder,'Images')


def cores():
	print(os.cpu_count())
	num_cores = os.cpu_count()

	manager = Manager()

	d = manager.dict()
	# image_path = '/home/sunin/Asun/DATA_for_STEPHAN/kinect/yolo_mask/'
	# l = os.listdir(image_path)
	input_list = range(0, num_data)

	input_data = np.array_split(input_list, num_cores-5)
	parmap.map(Synthesis,input_data)





def Synthesis(liss):

	num_object_images = np.zeros(len(obj_name), dtype=int)

	l = os.listdir(image_path)
	l_bg = os.listdir(bg_path)
	file_names = [x for x in l]
	for j in range(0,len(obj_name)):
		for i in file_names:
			if int(i[0:2]) == j+1:
				num_object_images[j] = num_object_images[j] + 1
	num_BG = len(l_bg)
	min_num_obj = 1
	max_num_obj = 3
####################
	# scale_min = 30
	# scale_max = 100
####################
	skip_count = 0
	for index in liss:
		print(index)
		BG_index = randint(1,num_BG)
		num_obj = randint(min_num_obj, max_num_obj)
		obj_index = np.zeros(num_obj, dtype=int)
		obj_picture_index = np.zeros(num_obj, dtype=int)
		BGRA = []
		object_names = []

		for i in range(0,num_obj):
			obj_index[i] = randint(0,len(obj_name)-1)
			obj_picture_index[i] = randint(1, num_object_images[obj_index[i]])
			BGRA.append(cv2.imread(image_path + '%.2i_%.4i.png'%(obj_index[i]+1, obj_picture_index[i]), cv2.IMREAD_UNCHANGED))
			object_names.append(obj_name[obj_index[i]])

		BG = cv2.imread(bg_path + '%i.jpg'%BG_index)
			# BG_seg = cv2.imread(background_folder_path + 'bg_seg.jpg')

		scale = np.zeros(num_obj,dtype=int)
		angle = np.zeros(num_obj,dtype=int)

		for i in range(0,num_obj):
			angle[i] = math.floor(360*random())-180
			if obj_index[i] == 3:
				angle[i] = math.floor(180*random())-180
			scale_percent = randint(85,115) # percent of original size
			width = int(BGRA[i].shape[1] * scale_percent / 100)
			height = int(BGRA[i].shape[0] * scale_percent / 100)

			dim = (width, height)

			BGRA[i] = cv2.resize(BGRA[i], dim)


			for i in range(0,num_obj):

			BGRA[i] = imutils.rotate_bound(BGRA[i],angle[i])
			index1 = cv2.findNonZero(BGRA[i][:,:,3])
			index1 = np.squeeze(index1)

			xmin_id = np.argmin(index1[:,0], axis=0) 
			xmax_id = np.argmax(index1[:,0], axis=0)
			ymin_id = np.argmin(index1[:,1], axis=0)
			ymax_id = np.argmax(index1[:,1], axis=0)

			xmin = index1[xmin_id,0]
			xmax = index1[xmax_id,0]
			ymin = index1[ymin_id,1]
			ymax = index1[ymax_id,1]

			BGRA[i] = BGRA[i][ymin:ymax,xmin:xmax,0:4]

		try :
			BG = cv2.resize(BG, (1280,720))
		except :
			print(bg_path + '%i.jpg'%BG_index)
			print("error")
			skip_count = skip_count +1
			continue
		BG_seg_sum = cv2.cvtColor(BG,cv2.COLOR_BGR2GRAY) * 0
		BG_seg = []
		for i in range(0,num_obj):
			BG_seg.append(cv2.cvtColor(BG,cv2.COLOR_BGR2GRAY) * 0)

		y_diff = np.zeros(num_obj, dtype=int)
		x_diff = np.zeros(num_obj, dtype=int)
		bbox_ymin = np.zeros(num_obj, dtype=int)
		bbox_xmin = np.zeros(num_obj, dtype=int)
		bbox_ymax = np.zeros(num_obj, dtype=int)
		bbox_xmax = np.zeros(num_obj, dtype=int)

		for i in range(0,num_obj):
			y_diff[i] = BG.shape[0] - BGRA[i].shape[0]
			x_diff[i] = BG.shape[1] - BGRA[i].shape[1]
			if y_diff[i] < 0:
				continue
			bbox_ymin[i] = math.floor(y_diff[i] * random())
			bbox_xmin[i] = math.floor(x_diff[i] * random())
			bbox_ymax[i] = bbox_ymin[i] + BGRA[i].shape[0]
			bbox_xmax[i] = bbox_xmin[i] + BGRA[i].shape[1]


		occur = 0
		for i in range(0,num_obj):
			if y_diff[i] < 0:
				occur = 1
				skip_count = skip_count +1
		if occur :
			continue

		alpha = []
		BGR = []
		for i in range(0,num_obj):
			alpha.append(BGRA[i][:,:,3])
			alpha[i] = alpha[i].astype(float)/255
			BGR.append(cv2.cvtColor(BGRA[i], cv2.COLOR_BGRA2BGR))

		try:
			BGR_array = np.array(BGR)
			alpha_array = np.array(alpha)
		except ValueError:
			print("error")
			skip_count = skip_count +1
			continue
		BG_array = np.array(BG)


		for i in range(0,num_obj):
			for j in range(0,3):				
				BG_array[bbox_ymin[i]:bbox_ymax[i], bbox_xmin[i]:bbox_xmax[i],j] = np.multiply(BGR_array[i][:,:,j],alpha_array[i]) + np.multiply(BG_array[bbox_ymin[i]:bbox_ymax[i], bbox_xmin[i]:bbox_xmax[i],j],(1-alpha_array[i]))
			BG_seg[i][bbox_ymin[i]:bbox_ymax[i], bbox_xmin[i]:bbox_xmax[i]] = alpha_array[i]
			BG_seg_sum = BG_seg_sum + BG_seg[i]

		error = 0
		for i in range(0,num_obj-1):
			pre_BG_seg_count = np.count_nonzero(BG_seg[i])
			for j in range(i+1,num_obj):
				BG_seg[i] = BG_seg[i] - BG_seg[i]*BG_seg[j]
				aft_BG_seg_count = np.count_nonzero(BG_seg[i])
				if ((aft_BG_seg_count/pre_BG_seg_count) <= 0.97):
					error=1

		if error:
			print("error occured")
			skip_count = skip_count +1
			continue


		for i in range(0,num_obj-1):
			for j in range(i+1,num_obj):
				BG_seg[i] = BG_seg[i] - BG_seg[i]*BG_seg[j]

		
		## DATA AUGMENTATION
		## brightness ##
		beta = randrange(-45,45)
		BG_array = cv2.addWeighted(BG_array, 1, np.zeros(BG_array.shape, BG_array.dtype),0, beta)
		## shadow ##
		eps = np.random.uniform()
		if eps > 0.7:
			BG_array = add_circle(BG_array)
		elif eps < 0.3:
			BG_array = add_shadow(BG_array)
		# ## noisy ##
		BG_array = add_noise(BG_array)
		# ## blur ##
		BG_array = add_blur(BG_array)
		# ## color tamperature ##
		BG_array = add_temp(BG_array)

		if YOLO_SAVE:
			f_label = open(yolo_label_save + '/%.6d.txt'%(index-skip_count),'w')
			cv2.imwrite(yolo_image_save + '/%.6d.jpg' % (index-skip_count), BG_array)
		
		
		margin = 15
		rand1 = randint(0,margin)
		rand2 = randint(0,margin)
		rand3 = randint(0,margin)
		rand4 = randint(0,margin)
		eps = random()

		for i in range(0,num_obj):
			f_ang_label = open(ang_label_save + '/%s_%.6d_%d.txt'%(obj_name[obj_index[i]],index-skip_count,i),'w')
			# cv2.imwrite(mask_save + '/%.6d_%.2d.png'%((index-skip_count),i+1),BG_seg[i])
			# f_annotation.write("%s\n"%object_names[i])
			temp_x_center = ((float(bbox_xmax[i])+float(bbox_xmin[i]))/2)/1280
			temp_y_center = ((float(bbox_ymax[i])+float(bbox_ymin[i]))/2)/720
			temp_width = (float(bbox_xmax[i])-float(bbox_xmin[i]))/1280
			temp_height = (float(bbox_ymax[i])-float(bbox_ymin[i]))/720

			if (obj_index[i] == 4) and (angle[i]>0):
				angle[i] -= 180

			if (eps >= 0.5):
				ang_save_img = BG_array[bbox_ymin[i]:bbox_ymax[i],bbox_xmin[i]:bbox_xmax[i],:]
			elif (eps >= 0.3):					
					a = max(bbox_ymin[i]-rand1,0)
					b = min(bbox_ymax[i]+rand2,720)
					c = max(bbox_xmin[i]-rand3,0)
					d = min(bbox_xmax[i]+rand4,1280)
					ang_save_img = BG_array[a:b,c:d,:]
			elif(eps < 0.3):				
				ang_save_img = BG_array[bbox_ymin[i]+rand1:bbox_ymax[i]-rand2,bbox_xmin[i]+rand3:bbox_xmax[i]-rand4,:]

			if ANG_SAVE:
				cv2.imwrite(ang_image_save + '/%s_%.6d_%d.jpg' % (obj_name[obj_index[i]],index-skip_count,i), ang_save_img)
				f_ang_label = open(ang_label_save + '/%s_%.6d_%d.txt'%(obj_name[obj_index[i]],index-skip_count,i),'w')				
				f_ang_label.write("{} {}".format(-angle[i],obj_index[i]+1))
				f_ang_label.close()

			if YOLO_SAVE:
				f_label.write("%i %f %f %f %f\n"%(obj_index[i],temp_x_center,temp_y_center,temp_width,temp_height))



		# f_annotation.close()
		if YOLO_SAVE:
			f_label.close()
		print(index)

	print("%s number of images are skiped"%skip_count)


if __name__=='__main__':
    cores()