import numpy as np
import os
from keras.optimizers import Adam, SGD
from keras.utils import plot_model
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Flatten, Input
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
import helper
import h5py
import cv2
from keras.preprocessing.image import *

class EmotionNet(object):
	"""docstring for EmotionNet"""
	def __init__(self):
		self.input_shape = None
		self.model = None
		self.epochs = 20
		self.threshold = 0.5
		self.model_path = "/root/Documents/Robo HR/models/CNN/"

	# def _create_vgg16model():
	# 	model = VGG16(include_top=True,weights='imagenet')
	# 	model.compile(optimizer=SGD(),loss = 'categorical_crossentropy',metrics = ['accuracy'])

	# 	return model

	# def img_to_features(self,img_path):
	# 	model = self._create_vgg16model()
	# 	img = load_img(path, target_size=(224,224))
	# 	x = img_to_array(img)
	# 	x = np.expand_dims(x,axis=0)
	# 	input = preprocess_input(x)

	# 	return model.predict(input)

	def create_model(self,input_shape):
		base_model = InceptionV3(weights='imagenet', include_top=False)
		x = base_model.output
		x = GlobalAveragePooling2D()(x)
		x = Dense(256,activation='relu')(x)
		x = Dropout(0.4)(x)
		x = Dense(units = 5,activation='softmax')(x)
		model = Model(base_model.input,x)

		for layer in base_model.layers:
			layer.trainable = False

		model.compile(loss = 'categorical_crossentropy',optimizer="rmsprop",metrics=['accuracy'])

		return model

	def fit(self,data_dir,epochs=None):
		if epochs is None:
			epochs = self.epochs
		self.model = self.create_model((224,224))
		train,validate,test = helper.image_agumenter('data/train/','data/test/',(224,224))
		self.model.fit_generator(train,
			epochs=epochs,
			steps_per_epoch=(train.samples//128),
			validation_data=validate,
			validation_steps=train.samples//128
		)

		#  Set first 249 layers not trainable and others as trainable it can be changed
		for layer in self.model.layers[:249]:
			layer.trainable = False
		for layer in self.model.layers[249:]:
			layer.trainable = True

		self.model.compile(optimizer=SGD(lr=0.001),loss = 'categorical_crossentropy')
		self.model.fit_generator(train,
			epochs=epochs,
			steps_per_epoch=(train.samples//128),
			validation_data=validate,
			validation_steps=train.samples//128
		)

		model_json = self.model.to_json()
		with open("m.json","w") as f:
			f.write(model_json)

		self.model.save_weights("weights.h5")
		self.config = dict()
		#self.config['input_shape'] = self.input_shape
		#self.config['labels'] = self.labels

		np.save('config.npy',self.config)
		
	def predict(self,img_path):
		classes = {
			0:'anger',
			1:'disgust',
			2:'fear',
			3:'happy',
			4:'neutral',
			5:'sad',
			6:'surprise'
		}

		img = load_img(img_path, target_size=(224,224))
		x = img_to_array(img)
		x = np.expand_dims(x,axis=0)
		input = preprocess_input(x)

		pred =  self.model.predict(input)
		k = max(pred[0])
		pred = [ i for i in range(len(pred[0])) if k == pred[0][i]]
		#top = decode_predictions(pred, top=1)[0]
		return classes[pred[0]]
		#eturn top[0].description,top[0].probability


	def load_model(self,model_path=None):
		if model_path is None:
			model_path = self.model_path
		with open(os.path.join(model_path,model_path+"m.json"),'r') as file:
			load_model = file.read()


		self.model = model_from_json(load_model)

		self.model.load_weights(os.path.join(model_path,model_path+"weights.h5"))
		plot_model(self.model,to_file="graph.png")

	def evalute(self,test_path):
		
		return self.model.evalute_generator(test)


def main():
	model = EmotionNet()
	#model.fit('/h',epochs=350)
	model.load_model()
	print(model.predict('489500_1970-05-05_2006.jpg'))
	#print(model.evalute('e'))

if __name__ == '__main__':
	main()
