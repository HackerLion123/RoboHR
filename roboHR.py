from models.CNN.model import EmotionNet
from models.oneshot.model import SiameseNet
from face_detection import extract_faces, get_face
import cv2
import time


def load_models():
	model1 = EmotionNet()
	model2 = SiameseNet()

	model1.load_model()
	model2.load_model()

	return model1,model2

def main():
 	model1,model2 = load_models()
 	#dataset = model2.create_model_encodings('models/oneshot/dataset')
 	#frame = cv2.imread('2018-05-07-131520.jpg')
 	#cv2.imshow('live',frame)
 	#cv2.waitKey(-1)
 	cap = cv2.VideoCapture("Jerry Seinfeld Stand-Up.mp4")
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
	 			img = clone[y-60:y+h+180,x-50:x+w+50]
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
		 			#cv2.imshow('live',frame_m)
		 		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
		 		font = cv2.FONT_HERSHEY_SIMPLEX
		 		open(name[1]+".csv","a").write("{0},{1}".format(time.ctime(),emotion))
		 		print(name)
		 		cv2.putText(frame,name[1] + "-"+ emotion,(x+w-int((x+w+1)/4),y+h+20),font,0.6,(0,255,0),2,cv2.LINE_AA)
		 		#print("hello")
		 		#out.write(frame)
		 		cv2.imshow('live',frame)
	 			cv2.waitKey(1)
	 			cv2.destroyAllWindows()	
	 	except Exception as e:
	 		print(e)

if __name__ == '__main__':
 	main()