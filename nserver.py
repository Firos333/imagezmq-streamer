# USAGE
#python webstreaming.py --ip  139.59.99.215 --port 8000
# python webstreaming.py --ip 0.0.0.0 --port 8000
#python webstreaming.py --ip  192.168.43.125 --port 8000
#139.59.99.215

# import the necessary packages
#from pyimagesearch.motion_detection import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
from datetime import datetime
import imutils
import time
import cv2
import imagezmq
import numpy as np
from imutils import build_montages
################### From server.py prev##########

# initialize the ImageHub object
#imageHub = imagezmq.ImageHub()

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
#CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
#	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
#	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
#	"sofa", "train", "tvmonitor"]

# load our serialized model from disk
#print("[INFO] loading model...")

# initialize the consider set (class labels we care about and want
# to count), the object count dictionary, and the frame  dictionary
#CONSIDER = set(["dog", "person", "car"])
#objCount = {obj: 0 for obj in CONSIDER}
#frameDict = {}

# initialize the dictionary which will contain  information regarding
# when a device was last active, then store the last time the check
# was made was now
#lastActive = {}
#lastActiveCheck = datetime.now()

# stores the estimated number of Pis, active checking period, and
# calculates the duration seconds to wait before making a check to
# see if a device was active
#ESTIMATED_NUM_PIS = 4
#ACTIVE_CHECK_PERIOD = 10
#ACTIVE_CHECK_SECONDS = ESTIMATED_NUM_PIS * ACTIVE_CHECK_PERIOD

# assign montage width and height so we can view all incoming frames
# in a single "dashboard"
#mW = args["montageW"]
#mH = args["montageH"]
#print("[INFO] detecting: {}...".format(", ".join(obj for obj in CONSIDER)))




#########################################################

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

# initialize the video stream and allow the camera sensor to
# warmup
#vs = VideoStream(usePiCamera=1).start()
#vs = VideoStream(src=0).start()


@app.route("/")
def index():
    global person_countlist,rpi_name,handlist
	# return the rendered template
    person= person_countlist
    rpi = rpi_name
    hands =handlist
    a_zip = zip(person,rpi,hands)

    return render_template("index.html",a_zip=a_zip)

# def detect_motion(frameCount):
def detect_motion(net,CLASSESS,frameDict):

    global outputFrame,lock,c
    while True:
            (rpiName, frame) = imageHub.recv_image()
            imageHub.send_reply(b'OK')
            if rpiName not in lastActive.keys():
                print("[INFO] receiving data from {}...".format(rpiName))
            # record the last active ime for the device from which we just
            # received a frame
            lastActive[rpiName] = datetime.now()
            #print("last",lastActive.keys())
            # resize the frame to have a maximum width of 400 pixels, then
            # grab the frame dimensions and construct a blob
            frame = imutils.resize(frame, width=400)
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)
            # pass the blob through the networkand obtain the detections and
            # predictions
            # print(net.setInput(blob),CLASSES)
            net.setInput(blob)
            detections = net.forward()
            # reset the object count for each object in the CONSIDER set
            objCount = {obj: 0 for obj in CONSIDER}
            # loop over the detections
            for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with
                # the prediction
                # print("in for")
                confidence = detections[0, 0, i, 2]
                # filter out weak detections by ensuring the confidence is
                # greater than the minimum confidence
                if confidence > args["confidence"]:
                    # extract the index of the class label from the
                    # detections
                    idx = int(detections[0, 0, i, 1])
                    # check to see if the predicted class is in the set of
                    # classes that need to be considered
                    if CLASSES[idx] in CONSIDER:
                        # increment the count of the particular object
                        # detected in the frame
                        objCount[CLASSES[idx]] += 1
                        # compute the (x, y)-coordinates of the bounding box
                        # for the object
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        # draw the bounding box around the detected object on
                        # the frame
                        cv2.rectangle(frame, (startX, startY), (endX, endY),(255, 0, 0), 2)
                            # draw the sending device name on the frame
            cv2.putText(frame, rpiName, (10, 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                # draw the object count on the frame
            label = ", ".join("{}: {}".format(obj, count) for (obj, count) in
                objCount.items())
            cv2.putText(frame, label, (10, h - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 2)
                # update the new frame in the frame dictionary

            if frame.size:
                hand_cascade = cv2.CascadeClassifier('Hand.Cascade.1.xml') 
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
                hands = hand_cascade.detectMultiScale(gray, 1.5, 5,minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE) 
                
                c=0
                for (x,y,w,h) in hands: 
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2) 
                    c+=1
                    cv2.putText(frame, f'P{c}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    if rpiName not in rpi_name:
                
                        handlist.append(c)
                    else:
                        
                        k=rpi_name.index(rpiName)
                        handlist[k]=c
                    
                    
       
            frameDict[rpiName] = frame
            if len(objCount) !=0:
                person_cnt=objCount['person']
                if rpiName not in rpi_name:
                    rpi_name.append(rpiName)
                    person_countlist.append(person_cnt)
                    handlist.append(c)
                else:
                    k=rpi_name.index(rpiName)
                    person_countlist[k] = person_cnt
                    handlist[k]=c

            # build a montage using images in the frame dictionary
            montages = build_montages(frameDict.values(), (w, h), (mW, mH))
            # acquire the lock, set the output frame, and release the
        # lock
    #	    with lock:
    #	        for (i, montage) in enumerate(montages):
    #		    outputFrame  = montage
            with lock:
                for (i, montage) in enumerate(montages):
                    # cv2.imshow("Home pet location monitor ({})".format(i),montage)
                            outputFrame = montage
            #                print("outframe",outputFrame,montage)
def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock
    # loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue
            # encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            # ensure the frame was successfully encoded
			if not flag:
				continue
            # yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),mimetype = "multipart/x-mixed-replace; boundary=frame")
# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	# ap = argparse.ArgumentParser()
	# ap.add_argument("-i", "--ip", type=str, required=True,
	# 	help="ip address of the device")
	# ap.add_argument("-o", "--port", type=int, required=True,
	# 	help="ephemeral port number of the server (1024 to 65535)")
	# ap.add_argument("-f", "--frame-count", type=int, default=32,
	# 	help="# of frames used to construct the background model")
	# args = vars(ap.parse_args())
    ########################### From server prev #############
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=True,help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.2,help="minimum probability to filter weak detections")
    ap.add_argument("-mW", "--montageW", required=True, type=int,help="montage frame width")
    ap.add_argument("-mH", "--montageH", required=True, type=int,help="montage frame height")
    args = vars(ap.parse_args())
	# start a thread that will perform motion detection
	# t = threading.Thread(target=detect_motion, args=(
	# 	args["frame_count"],))
   # initialize the ImageHub object
    imageHub = imagezmq.ImageHub()
    global person_countlist,rpi_name,handlist
    person_countlist=[]
    rpi_name=[]
    handlist=[]
# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the consider set (class labels we care about and want
# to count), the object count dictionary, and the frame  dictionary
    CONSIDER = set(["person"])
    objCount = {obj: 0 for obj in CONSIDER}
    frameDict = {}

# initialize the dictionary which will contain  information regarding
# when a device was last active, then store the last time the check
# was made was now
    lastActive = {}
    lastActiveCheck = datetime.now()

# stores the estimated number of Pis, active checking period, and
    # calculates the duration seconds to wait before making a check to
    # see if a device was active
    ESTIMATED_NUM_PIS = 4
    ACTIVE_CHECK_PERIOD = 10
    ACTIVE_CHECK_SECONDS = ESTIMATED_NUM_PIS * ACTIVE_CHECK_PERIOD

    # assign montage width and height so we can view all incoming frames
    # in a single "dashboard"
    mW = args["montageW"]
    mH = args["montageH"]
    print("[INFO] detecting: {}...".format(", ".join(obj for obj in CONSIDER)))
    t = threading.Thread(target=detect_motion,args=(net,CLASSES,frameDict,))
    t.daemon = True
    t.start()
    app.run(host="128.199.20.246", port=8000, debug=True,threaded=True, use_reloader=False)
	# start the flask app
	# app.run(host=args["ip"], port=args["port"], debug=True,
	# 	threaded=True, use_reloader=False)

# release the video stream pointer
#vs.stop()
