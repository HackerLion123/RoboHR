from __future__ import division, print_function

import random
import os
import cv2
import tensorflow as tf
import numpy as np

from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, Lambda, Dropout
from keras.optimizers import Adam, SGD, RMSprop
from keras.preprocessing.image import img_to_array,array_to_img,ImageDataGenerator,load_img
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import keras.backend as k


def triplet_loss(y_true, y_pred):
	alpha = 0.002
	print(y_true,y_pred)
	anchor, positive, negative = y_true, y_pred[0], y_pred[1]


	positive_dist = k.sum(k.square(anchor - positive),axis = 0)

	negative_dist = k.sum(k.square(anchor - negative),axis = 0)

	basic_loss = positive_dist - negative_dist + alpha

	loss = k.sum(k.maximum(basic_loss,0))

	return loss


def eulidean_distance(vect):
	x, y = vect

	return k.sqrt(k.maximum(k.sum(k.square(x - y),axis = 1,keepdims=True),k.epsilon()))

def eulidean_output_shape(shapes):
	shape1, shape2 = shapes
	return (shape1[0],1)

def contrastive_loss(y_true, y_pred):
		margin = 1

		return k.mean(y_true * k.square(y_pred) 
			+ (1 - y_true) * k.square(k.maximum(margin - y_pred, 0)))

def cosine_similarity_loss(y_true, y_pred):
	Ls = contrastive_loss(y_true, y_pred)

class SiameseNet(object):
	"""docstring for SiameseNet"""
	def __init__(self):
		
		self.model = None
		self.model_path = 'models/oneshot/'
		self.epochs = 20
		self.labels = None
		self.threshold = 0.5
		self.input_shape = None
		self.vgg16_model = None
		self.vgggraph = None
		self.database = None
		self.graph = tf.get_default_graph()

	def load_model(self,model_path = None):
		if model_path is None:
			model_path = self.model_path
		# path = os.path.join("/root/Documents/Robo HR",model_path)
		path = os.getcwd()
		self.config = np.load(os.path.join(path,'config.npy')).item()
		self.database = np.load(os.path.join(path,'database.npy')).item()
		self.labels = self.config['labels']
		self.input_shape = self.config['input_shape']

		self.vgg16_model = self.create_vgg16_model()
		self.model = self.create_network(shape=self.input_shape)
		self.model.load_weights(os.path.join(path,'weight.h5'))
		self.graph = tf.get_default_graph()

	def img_to_encoding(self,path):
		if self.vgg16_model is None:
			self.vgg16_model = self.create_vgg16_model()

		image = cv2.imread(path,1)

		# shape (224,224) for vgg16 model
		img =cv2.resize(image,(224,224), interpolation=cv2.INTER_AREA)
		input = img_to_array(img)
		input = np.expand_dims(input, axis = 0)
		input = preprocess_input(input)
		with self.graph.as_default():
			return self.vgg16_model.predict(input)


	def create_pairs(self, dataset, names):
	 	num_classes = len(dataset) 
	 	pairs = []
	 	labels = []
	 	n = min([len(dataset[name]) for name in dataset])
	 	for m in range(len(names)):
	 		name = names[m]
	 		x = dataset[name]
	 		for i in range(n):
	 			pairs += [[x[i], x[(i+1)%n]]]
	 			inc = random.randrange(1,num_classes)
	 			dn = (m + inc) % num_classes
	 			z1, z2 = x[i], dataset[names[dn]][i]
	 			pairs  += [[z1, z2]]
	 			labels += [1,0]

	 	print(np.array(pairs).shape)
	 	print(np.array(labels).shape)
	 	return np.array(pairs), np.array(labels)

	def create_vgg16_model(self):
		"""
			VGG16 model is used to extract features. It has the dense softmax layer
			when top=True
		"""
		model = VGG16(include_top=True,weights='imagenet')
		model.compile(optimizer=SGD(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

		return model

	def create_base_network(self,input_shape):
		input = Input(shape=input_shape)
		x = Flatten()(input)
		# x = Dense(256, activation='relu')(x)
		# x = Dropout(0.4)(x)
		# x = Dense(512, activation='relu')(x)
		# x = Dropout(0.5)(x)
		x = Dense(512, activation='softmax')(x)

		return Model(input, x)

	def create_network(self,shape):
		base_network = self.create_base_network(shape)

		input_a = Input(shape=shape)
		input_b = Input(shape=shape)

		processed_a = base_network(input_a)
		processed_b = base_network(input_b)

		distance = Lambda(eulidean_distance, 
			output_shape=eulidean_output_shape)([processed_a, processed_b])

		model = Model([input_a, input_b], distance)

		rms = RMSprop(lr=0.002)

		model.compile(loss=contrastive_loss, optimizer = rms, metrics=[self.accuracy])

		print(model.summary())

		return model

	def create_model_encodings(self,dir,augment=True,save_encodings=False):
		database = {}
		current_dir = os.getcwd()
		dir = os.path.join(current_dir,dir)
		for file in os.listdir(dir):
			if file not in database:
				database[file] = []
			os.chdir(os.path.join(dir,file))
			if augment is True: 
				#Augment the images
				datagen = ImageDataGenerator(
					rescale = 1./255,
					shear_range = 0.2,
					zoom_range = 0.2,
					width_shift_range=0.1,
					height_shift_range=0.1,
					horizontal_flip=True,
					vertical_flip=False
				)
				b = 0
				for e in os.listdir('.'):
					e = e.split('.')
					if len(e)>2:
						b = 1
						break
				for e in os.listdir('.'):
					if b == 1:
						break
					count = 3
					img = img_to_array(load_img(e))
					output_path = e+"{}.jpg"
					img = img.reshape((1,) + img.shape)

					images = datagen.flow(img, batch_size=2)
					for i, new_img in enumerate(images):

						new_img = array_to_img(new_img[0], scale=True)
						new_img.save(output_path.format(i+1))
						if i>=count:
							break

			for e in os.listdir('.'):
				database[file].append(self.img_to_encoding(e))

		os.chdir(current_dir)
		if save_encodings:
			np.save('database.npy',database)
		return database


	def fit(self,database=None,model_dir=".",epochs=None,batch_size=None,threshold=None):
		if batch_size is  None:
			batch_size = 32
		if epochs is not None:
			self.epochs = epochs
		if database is None:
			database = self.database

		for name, feature in database.items():
			self.input_shape = feature[0].shape
			break

		self.vgg16_model = self.create_vgg16_model()

		self.model = self.create_network(shape=self.input_shape)

		with open('arch.h5','w') as f:
			f.write(self.model.to_json())

		names = []
		self.labels = dict()
		for name in database:
			names.append(name)
			self.labels[name] = len(self.labels)

		self.config = dict()
		self.config['input_shape'] = self.input_shape
		self.config['labels'] = self.labels

		np.save('config.npy',self.config)

		checkpoint = ModelCheckpoint('weights.h5',monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True)
		reducelr = ReduceLROnPlateau(monitor='val_loss',factor=0.5,verbose=1,paitence=100,min_lr=0.00001)

		t_x, t_y = self.create_pairs(database, names)
		print(t_x.shape)
		#print([t_x[:,0],t_x[:,1]])
		self.model.fit([t_x[:,0],t_x[:,1]], t_y,
			batch_size=batch_size,
			epochs=self.epochs,
			validation_split=0.2,
			verbose=1,
			callbacks=[checkpoint,reducelr]
		)
		self.model.save_weights("weight.h5")


	def accuracy(self, y_true, y_pred):
		print(k.cast(y_pred < self.threshold, y_true.dtype))
		return k.mean(k.equal(y_true, k.cast(y_pred < self.threshold, y_true.dtype)))

	def evalute(self,test,labels):
		test = self.create_model_encodings('val/',augment=False)
		names = []
		labels = dict()
		for name in test:
			names.append(name)
			labels[name] = len(labels)
		t_x, t_y = self.create_pairs(test, names)
		#print("hello")
		print(self.model.evaluate([t_x[:,0],t_x[:,1]],t_y,verbose=1))

	def predict(self,img_path, dataset=None):
		if dataset is None:
			dataset = self.database

		encoding = self.img_to_encoding(img_path)

		#Minimum distance
		min_dist = 150
		identity = None

		for name,img in dataset.items():

			input_pairs = []
			for i in range(len(img)):
				input_pairs.append([encoding,img[i]])
			input_pairs = np.array(input_pairs)
			with self.graph.as_default():
				dist = self.model.predict([input_pairs[:,0],input_pairs[:,1]])
			dist = np.average(dist,axis=1)[0]
			if dist < min_dist:
				min_dist = dist
				identity = name

		if min_dist > self.threshold:
			identity = "unknown"

		return min_dist,identity


def main():
	model = SiameseNet()
	
	# database = model.create_model_encodings(os.path.join(os.getcwd(),'dataset/'),save_encodings=True)
	model.fit(database=np.load('database.npy').item(),epochs=9000)
	model.load_model()
	
	print(model.predict('file15.jpg'))
	# print(model.predict('file1.jpg'))
	print(model.predict('sam3.jpeg2.jpg'))
	# print(model.predict('17.jpeg'))
	print(model.evalute("",""))



if __name__ == '__main__':
	main()