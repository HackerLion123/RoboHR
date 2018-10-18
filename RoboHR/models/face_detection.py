import os
import argparse
import cv2


def extract_faces(frame):

	#frame = cv2.equalizeHist(frame)

	face_cascade =   cv2.CascadeClassifier('/root/Documents/Robo HR (copy)/RoboHR/models/haarcascades/haarcascade_frontalface_alt.xml')
	profile_cascade = cv2.CascadeClassifier('/root/Documents/Robo HR (copy)/RoboHR/models/haarcascades/haarcascade_profileface.xml')

	faces = face_cascade.detectMultiScale(frame,1.3,5)
	profiles = profile_cascade.detectMultiScale(frame,1.3,5)

	return faces
	
		

def get_face(frame,face_points):
	count = 0
	print(face_points)
	clone = frame.copy()
	features = []
	for x,y,w,h in face_points:
		print("dick dick..........")
		img = clone[y:y+h+10,x:x+w+10]
		cv2.imshow('live',img)
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
	return features

# def extract_faces(frame):

# 	frame = cv2.equalizeHist(frame)

# 	face_cascade =   cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
# 	profile_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_profileface.xml')

# 	faces = face_cascade.detectMultiScale(frame,1.3,5)
# 	profiles = profile_cascade.detectMultiScale(frame,1.3,5)
# 	features = []
# 	try:
# 		clone = frame.copy()
# 		for (x,y,w,h) in faces:

# 			img = clone[y-20:y+h+20,x-10:x+w+10]
# 			#print(img)
# 			try:
# 				cv2.imshow('live',img)
# 				features.append(img)
# 			except:
# 				pass
# 	except:
# 		pass

# 	return features
	
def main():

	cap = cv2.VideoCapture(0)
	

	# os.chdir("face I/")
	# for img in os.listdir():
	#  	frame = cv2.imread(img)	
	#  	extract_faces(frame)
	# count = 0
	# while True:
	# 	ret,frame = cap.read()
	# 	# gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	# 	count += 1
	# 	extract_faces(frame,count)


if __name__ == '__main__':
	main()

