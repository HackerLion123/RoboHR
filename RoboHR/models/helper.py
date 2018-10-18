import os
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
#from face_detection import * 

def load_images(source):

	img_path = os.path.join(source)

	return cv2.imread(img_path)

def image_agumenter(train,test,shape):
	datagen = ImageDataGenerator(
			rescale = 1./255,
			shear_range = 0.2,
			zoom_range = 0.2,
			width_shift_range=0.1,
			height_shift_range=0.1,
			horizontal_flip=True,
			vertical_flip=False
		)

	test_gen = ImageDataGenerator(
			rescale = 1./255,
			horizontal_flip=True,
			vertical_flip=False
		)
	train_set = datagen.flow_from_directory(train,target_size=shape,batch_size=32, color_mode='rgb',class_mode='categorical',shuffle=True)

	#validate = datagen.flow_from_directory(validate,target_size=shape,batch_size=1,class_mode='categorical',shuffle=True)

	test_set = test_gen.flow_from_directory(test,target_size=shape,batch_size=1,color_mode='rgb',class_mode='categorical',shuffle=False)

	return train_set,test_set
def load_data(src):
	os.chdir(src)
	files = os.listdir('.')
	n = len(files)

	# Navie Method
	# for fpath in fpaths:
	#   img = cv2.imread(fpath, cv2.CV_LOAD_IMAGE_COLOR)
	#   image_list.append(img)

	# data = np.vstack(image_list)
	# data = data.reshape(-1, 256, 256, 3)
	# data = data.transpose(0, 3, 1, 2)


	# Better Way
	data = np.empty((n,3,32,30),dtype=np.uint8)
	for i,f in enumerate(files):
		img = cv2.imread(f)
		data[i,...] = img.transpose(2,0,1)
	return data


def load_dataset():
	pass

if __name__ == '__main__':
	data = load_data('/root/Documents/Robo HR/faces/at33')
	cv2.imshow('hi',data[0])
	
	
