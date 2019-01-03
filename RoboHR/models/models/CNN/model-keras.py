import numpy as np
from keras.optimizers import Adam, SGD
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import helper
import h5py
from keras.preprocessing.image import *


def createModel():
	model = Sequential()
	
	model.add(Conv2D(32,(5,5),input_shape = (299, 299, 3),activation="relu"))
	model.add(MaxPooling2D(pool_size = (2, 2)))

	model.add(Conv2D(64,(7,7),activation="relu"))
	model.add(MaxPooling2D(pool_size = (2, 2)))

	model.add(Flatten())

	model.add(Dense(units = 256,activation='sigmoid'))
	model.add(Dropout(0.4))
	model.add(Dense(units = 7,activation='softmax'))

	model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics = ['accuracy'])
	return model

def predict(model,img):
	classes = {
		0:'happy',
		1:'sad',
		2:'anger',
		3:'surprise',
		4:'disgust',
		5:'fear',
		6:'neutral'
	}

	result = model.predict(img)

	return classes[result]


def load_model(file):
	with open("model.json",'r') as file:
		load_model = file.read()

	load_model = model_from_json(load_model)

	load_model.load_weights("model.h5")

	return load_model

def main():

	model = createModel()

	train,test = helper.image_agumenter('data/train/','data/test/',(299,299))
	print(train)
	#Train the model
	model.summary()
	model.fit_generator(train,
		steps_per_epoch=10,
		epochs=40
	)

	

	#Evalute the model
	#score = model.evalute_generator(test)

	model_json = model.to_json()
	with open("model.json","w") as f:
		f.write(model_json)

	model.save_weights("model.h5")

	#print(predict(model,''))


if __name__ == '__main__':
	main()
