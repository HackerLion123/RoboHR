import numpy as np
import cv2
import os

def check_model(img):

	frame_m = cv2.imread(img)

	frame = cv2.cvtColor(frame_m,cv2.COLOR_BGR2GRAY)
	
	frame = cv2.equalizeHist(frame)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	frame = clahe.apply(frame)

	
	face_cascade =   cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
	profile_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_profileface.xml')

	faces = face_cascade.detectMultiScale(frame,1.2,10)
	# faces_alt = face_alt_cascade.detectMultiScale(gray,1.3,5)
	profiles = profile_cascade.detectMultiScale(frame,1.2,10)

	print(len(faces))
	print(len(profiles))

	for (x,y,w,h) in faces:
		cv2.rectangle(frame_m,(x,y),(x+w,y+h),(0,255,0),1)

	for (x,y,w,h) in profiles:
		cv2.rectangle(frame_m,(x,y),(x+w,y+h),(0,255,0),1)
	cv2.imshow('live',frame_m)
	cv2.waitKey(5000)
	cv2.destroyAllWindows()
	
def main():
	check_model('/root/Pictures/2018-05-07-131520.jpg')
	# os.chdir("face I/")
	# for img in os.listdir():
	# 	print(img)
	# 	check_model(img)

if __name__ == '__main__':
	main()

