from __future__ import division,print_function
from helper import load_data
import numpy as np
import tensorflow as tf


class CNN:
	"""docstring for EmotionDetector"""
	def __init__(self):
		pass
		
	def add_conv(self,shape,filters,kernel,padding):
		return tf.layers.conv2d(
			inputs=input_layer,
			filters=filters,
			kernel_size=kernel,
			padding="same",
			activation=tf.nn.relu)

	def add_flatlayer(self):
		pass

	def add_pool(self,input):
		return tf.layers.max_pooling2d()

	def add_dense(self,shape):
		pass

def build_model():
	classifier = CNN()
	classifier.add_conv()
	return classifier



def load_model():
	pass


if __name__ == '__main__':
	main()

