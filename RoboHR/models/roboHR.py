from .models.CNN.model import EmotionNet
from .models.oneshot.model import SiameseNet
from .face_detection import extract_faces, get_face
import cv2
import os
import base64
from PIL import Image
from io import BytesIO
from urllib.request import urlretrieve
import requests
import time



def load_models():
	model1 = EmotionNet()
	model2 = SiameseNet()

	model1.load_model()
	model2.load_model()

	return model1,model2

model1,model2 = load_models()

def get_image_from_url(url):
	try:
		print("works")
		#response = requests.get(url,stream=True)
		open("/root/Documents/Robo HR (copy)/RoboHR/model/static/images/new.png","wb").write(base64.decodebytes(url))

		return cv2.imread("/root/Documents/Robo HR (copy)/RoboHR/model/static/images/new.png")

	except Exception as e:
		print("get_image_from_url ",e)

def read_image(img):
	frame = get_image_from_url(img)
	frame = cv2.imread("/root/Documents/Robo HR (copy)/RoboHR/model/static/images/new.png")
	#print(frame)
	faces = extract_faces(frame)
	print("It also works")
	try:
 		#print("hello")
 		clone = frame.copy()
 		frame_m = frame.copy()
 		print(faces)
 		for (x,y,w,h) in faces:
 			#print(faces)
 			#face = get_face(clone,(x,y,w,h))
 			count = 0
 			features = []
 			#print("tick tick..........")
 			img = clone[y-50:y+h+50,x-50:x+w+50]
 			#cv2.imshow('live',img)
 			#print(img)
 			try:
 				print("hello") 
 				count += 1
 				file_name = '/root/Documents/Robo HR (copy)/RoboHR/model/static/images/file{}.jpg'.format(count)
 				features.append(file_name)
 				cv2.imwrite(file_name,img)
 			except Exception as e:
 				print("face",e)
 				#return "no faces Detected","none"
 			#print(features)
 			for f in features:
 				print(f)
 				emotion = model1.predict(f)
 				print("emo")
	 			name = model2.predict(f)
	 			# print(name)
	 			# print(emotion)
	 			#cv2.imshow('live',frame_m)
	 		# cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
	 		# font = cv2.FONT_HERSHEY_SIMPLEX
	 		file = "{}.csv".format(name[1])
	 		print(name)
	 		file_name = os.path.join("/root/Documents/Robo HR (copy)/RoboHR/models/","{}.csv".format(name[1]))
	 		if file not in os.listdir("/root/Documents/Robo HR (copy)/RoboHR/models"):
	 		 	open(file_name,"w").write("DateTime,Emotion\n")
	 		open(file_name,"a").write("{0},{1}\n".format(time.ctime(),emotion))
	 		return name[1],emotion
	 		# cv2.putText(frame,name[1] + "-"+ emotion,(x+w-int((x+w+1)/4),y+h+20),font,0.6,(0,255,0),2,cv2.LINE_AA)
	 		#print("hello")
	 		#out.write(frame)
	 		# cv2.imshow('live',frame)
 			# cv2.waitKey(500)
 			# cv2.destroyAllWindows()
	except Exception as e:
 		print("read_image",e)
 		return "no faces Detected","none"


def main():
 	#model1,model2 = load_models()
 	#dataset = model2.create_model_encodings('models/oneshot/dataset')
 	#frame = cv2.imread('2018-05-07-131520.jpg')
 	#cv2.imshow('live',frame)
 	#cv2.waitKey(-1)
 	cap = cv2.VideoCapture(0)
 	#fourcc = cv2.VideoWriter_fourcc(*'XVID')
 	#out = cv2.VideoWriter('output.avi',fourcc,1.0,(640,480))
 	while True:
 		re,frame = cap.read()
	 	faces = extract_faces(frame)
	 	try:
	 		#print("hello")
	 		clone = frame.copy()
	 		frame_m = frame.copy()
	 		#print(faces)
	 		for (x,y,w,h) in faces:
	 			#print(faces)
	 			#face = get_face(clone,(x,y,w,h))
	 			count = 0
	 			features = []
	 			#print("tick tick..........")
	 			img = clone[y-50:y+h+50,x-50:x+w+50]
	 			#cv2.imshow('live',img)
	 			#print(img)
	 			try: 
	 				count += 1
	 				file_name = 'file{}.jpg'.format(count)
	 				features.append(file_name) 
	 				cv2.imwrite('file{}.jpg'.format(count),img)
	 				#cv2.imshow('face',img)
	 				#cv2.waitKey(5)
	 			except Exception as e:
	 				print(e)
	 			print(features)
	 			for f in features:
	 				emotion = model1.predict(f)
		 			name = model2.predict(f)
		 			print(name)
		 			print(emotion)
		 			#cv2.imshow('live',frame_m)
		 		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
		 		font = cv2.FONT_HERSHEY_SIMPLEX
		 		file_name = "{}.csv".format(name[1])
		 		if file_name not in os.listdir('.'):
		 			open(file_name,"a").write("DateTime,Emotion\n")	
		 		open(file_name,"a").write("{0},{1}\n".format(time.ctime(),emotion))
		 		cv2.putText(frame,name[1] + "-"+ emotion,(x+w-int((x+w+1)/4),y+h+20),font,0.6,(0,255,0),2,cv2.LINE_AA)
		 		#print("hello")
		 		#out.write(frame)
		 		cv2.imshow('live',frame)
	 			cv2.waitKey(500)
	 			cv2.destroyAllWindows()	
	 	except Exception as e:
	 		print(e)

if __name__ == '__main__':
 	#main()
 	read_image("h")