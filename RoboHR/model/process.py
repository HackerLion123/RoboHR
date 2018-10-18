import pandas as pd
from pandas import Series
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


base_dir = "/root/Documents/Robo HR (copy)/RoboHR/models/"

def get_barchart(name):
	data = pd.read_csv(base_dir+name)
	ax = data['Emotion'].value_counts().plot.bar()
	fig = ax.get_figure()
	fig.savefig('/root/Documents/Robo HR (copy)/RoboHR/model/static/images/1.png')

	return "/static/images/1.png"

def get_graph(name):
	class_values = {
		'happy':4,
		'sad':-2,
		'neutral':1,
		'normal':0,
		'anger':-4,
		'fear':-2
	}

	data = pd.read_csv(base_dir+name)
	x = data['DateTime']
	y = [   class_values[emo]  for emo in data['Emotion'] ]
	fig1 = plt.figure()
	plt.plot(x,y)
	# plt.xlabel('DateTime')
	# plt.ylabel('Emotion')
	# series = Series([x,y],header=0)
	# ax1 = series.hist()
	#fig1 = ax1.get_figure()
	fig1.savefig('/root/Documents/Robo HR (copy)/RoboHR/model/static/images/18.png')

	return "/static/images/18.png"


	



