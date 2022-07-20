# USAGE
# python detect_mask_video.py

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from playsound import playsound
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import serial
import sys

PORT = 'com6'
BAUDRATE = 9600


from threading import Thread 
#from plyer import notification

def sendStartToArduino():
	global isSend
	arduinoSerial.write('1\n'.encode())
	isSend = 0;

def sendStopToArduino():
	global isSend
	arduinoSerial.write('0\n'.encode())
	isSend = 0;

class CountdownThread(Thread): 
	def __init__(self, n, callback1, callback2): 
		super().__init__() 
		self.n = n
		self.cb1 = callback1
		self.cb2 = callback2
	
	def run(self): 
		while self.n > 0:
			if(self.n < 2):
				self.cb1()
			self.n -= 1
			time.sleep(1)

	def stop(self):
		self.n = 0
		self.cb2()

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			try :
				# compute the (x, y)-coordinates of the bounding box for
				# the object
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# ensure the bounding boxes fall within the dimensions of
				# the frame
				(startX, startY) = (max(0, startX), max(0, startY))
				(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

				# extract the face ROI, convert it from BGR to RGB channel
				# ordering, resize it to 224x224, and preprocess it
				face = frame[startY:endY, startX:endX]
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)

				# add the face and bounding boxes to their respective
				# lists
				faces.append(face)
				locs.append((startX, startY, endX, endY))
			except Exception as e :
				print(e)

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

if __name__ == '__main__':
	arduinoSerial = serial.Serial(PORT,BAUDRATE)
	isSend = 0;

	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--face", type=str,
		default="face_detector",
		help="path to face detector model directory")
	ap.add_argument("-m", "--model", type=str,
		default="mask_detector.model",
		help="path to trained face mask detector model")
	ap.add_argument("-c", "--confidence", type=float, default=0.5,
		help="minimum probability to filter weak detections")
	args = vars(ap.parse_args())

	# load our serialized face detector model from disk
	print("[INFO] loading face detector model...")
	prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
	weightsPath = os.path.sep.join([args["face"],
		"res10_300x300_ssd_iter_140000.caffemodel"])
	faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

	# load the face mask detector model from disk
	print("[INFO] loading face mask detector model...")
	maskNet = load_model(args["model"])

	# initialize the video stream and allow the camera sensor to warm up
	print("[INFO] starting video stream...")

	# ganti link video stream dari cctv / dari webcam lainnya di bawah ini
	# isi dengan 0 jika pakai default punya laptop
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

	# loop over the frames from the video stream
	while True:
		# grab the frame from the threaded video stream and resize it
		# to have a maximum width of 400 pixels
		frame = vs.read()
		frame = imutils.resize(frame, width=1024)

		# detect faces in the frame and determine if they are wearing a
		# face mask or not
		(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

		# loop over the detected face locations and their corresponding
		# locations
		for (box, pred) in zip(locs, preds):
			# unpack the bounding box and predictions
			(startX, startY, endX, endY) = box
			(mask, withoutMask) = pred

			# determine the class label and color we'll use to draw
			# the bounding box and text
			label = "Pakai Masker " if mask > withoutMask else "Tidak Pakai Masker "
			color = (0, 255, 0) if label == "Pakai Masker " else (0, 0, 255)
			c = None
			if label == "Pakai Masker ":
				if isSend == 0 :
					isSend = 1
					try :
						c = CountdownThread(2, sendStartToArduino, sendStopToArduino) 
						c.start()
					except Exception as e :
						print(e)
			else :
				if c != None :
					c.stop()

			# 	notification.notify(
			# 		title = "***No Mask Detected***",
	  #               		message = "Wear Mask to stay safe! ",
	  #               		app_icon = "images/1.ico",    #ico file should be downloaded
	  #               		timeout = 1
	  #           		)
				
				
			# include the probability in the label
			label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

			# display the label and bounding box rectangle on the output
			# frame
			cv2.putText(frame, label, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

			# Alarm when "No Mask" detected
			# if mask < withoutMask:
			# 	path = os.path.abspath("Alarm.wav")
			# 	playsound(path)

		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()

