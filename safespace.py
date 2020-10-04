import cv2
import argparse
import numpy as np
import imutils
import time
import dlib
import math
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS

# user variables
videopath = './example_video.mp4' # PATH TO INPUT VIDEO
outputpath = './output_video.mp4' # PATH TO OUTPUT VIDEO
prototxt = './mobilenet_ssd/MobileNetSSD_deploy.prototxt'
modelpath = './mobilenet_ssd/MobileNetSSD_deploy.caffemodel'
conf = 0.4 # confidence value
skip_frames = 30

# capacity
cur_inside = 0
max_capacity = 10 

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, modelpath)

# get video file
print("[INFO] opening video file...")
vs = cv2.VideoCapture(videopath)

# video writer
writer = None

# frame dimensions
W = None
H = None

ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

# number of people that have entered or exited 
outgoing = 0
incoming = 0

# start the frames per second throughput estimator
totalFrames = 0
fps = FPS().start()

# loop over frames from the video stream
while True:
    too_close = 0 # keeps track of how many people are standing within 2 metres of each other 
    
    # grab the next frame and handle if we are reading from either VideoCapture or VideoStream
    frame = vs.read()
    frame = frame[1] 

    # if we are viewing a video and we did not grab a frame then we have reached the end of the video
    if videopath is not None and frame is None:
        break

    # resize the frame to have a maximum width of 700 pixels (the
    # less data we have, the faster we can process it), then convert
    # the frame from BGR to RGB for dlib
    frame = imutils.resize(frame, width=700)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # if the frame dimensions are empty, set them
    if W is None or H is None:
        (H, W) = frame.shape[:2]
        
    # if we are supposed to be writing a video to disk, initialize the writer
    if outputpath is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(outputpath, fourcc, 30, (W, H), True)
    
    # initialize our list of bounding box rectangles returned by either (1) our object detector 
    # or (2) the correlation trackers
    rectangles = []

    # check to see if we should run a more computationally expensive
    # object detection method to aid our tracker
    if totalFrames % skip_frames == 0:
        # initialize our new set of object trackers
        trackers = []

        # convert the frame to a blob and pass the blob through the
        # network and obtain the detections
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        net.setInput(blob)
        detections = net.forward()
        
        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by requiring a minimum confidence
            if confidence > conf:
                # compute the (x, y)-coordinates of the bounding box for the object
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                # construct a dlib rectangle object from the bounding
                # box coordinates and then start the dlib correlation tracker
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
                tracker.start_track(rgb, rect)

                # add the tracker to our list of trackers so we can utilize it during skip frames
                trackers.append(tracker)
                
    # otherwise, we should utilize our object *trackers* rather than
    # object *detectors* to obtain a higher frame processing throughput
    else:
        for tracker in trackers:
            # update the tracker and grab the updated position
            tracker.update(rgb)
            pos = tracker.get_position()

            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # add the bounding box coordinates to the rectangles list
            rectangles.append((startX, startY, endX, endY))
    
    # draw a horizontal line in the center of the frame -- once an
    # object crosses this line we will determine whether they are entering or exiting    
    cv2.line(frame, (0, H // 8), (W, H // 8), (102, 102, 0), 2) #cv2.line(frame, (W // 2, 0), (W // 2, H), (0, 255, 255), 2)
    
    # use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids
    objects = ct.update(rectangles)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        
        # check to see if a trackable object exists for the current object ID
        to = trackableObjects.get(objectID, None)

        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)

        # otherwise, there is a trackable object so we can utilize it to determine direction
        else:
            # the difference between the y-coordinate of the *current*
            # centroid and the mean of *previous* centroids will tell
            # us in which direction the object is moving (negative for
            # 'up' and positive for 'down')
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)
            
            # check to see if the object has been counted or not
            if not to.counted:
                # if the direction is negative (indicating the object
                # is moving up) AND the centroid is above the center
                # line, count the object
                # moving up across the line indicates exiting (decrease current capacity)
                if direction < 0 and centroid[1] < H // 8:
                    cur_inside -= 1                   
                    to.counted = True

                # if the direction is positive (indicating the object
                # is moving down) AND the centroid is below the
                # center line, count the object
                # moving down across the line indicates entering (increase current capacity)
                elif direction > 0 and centroid[1] > H // 8:
                    cur_inside += 1
                    to.counted = True
                    
        # store the trackable object in our dictionary
        trackableObjects[objectID] = to

        # draw both the ID of the object and the centroid of the object on the output frame 
        text = "ID {}".format(objectID)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 153, 76), -1)

        # checking if the other objects are too close to the current object 
        for (objectID_2, centroid_2) in objects.items():
            if objectID != objectID_2:
                dist = math.sqrt((centroid[0]-centroid_2[0])**2+(centroid[1]-centroid_2[1])**2)
                if (dist < 125):
                    too_close += 1
                    cv2.line(frame, (centroid[0], centroid[1]), (centroid_2[0], centroid_2[1]), (102, 102, 255), 2)
                
        
    # updating screen messages
    if (too_close == 0):
        risk = "None"
        message = "SAFE - DISTANCE MAINTAINED"
    elif (too_close <= 3):
        risk = "Low"
        message = "CLOSE - DISTANCE MOSTLY MAINTAINED"
    elif (too_close <= 5):
        risk = "Moderate"
        message = "CLOSE - DISTANCE BARELY MAINTAINED"
    else:
        risk = "High"
        message = "EXTREMELY CLOSE - DISTANCE NOT MAINTAINED"
        
    if (cur_inside > max_capacity):
        info = [("Status", message), ("Current capacity", cur_inside), ("Risk level", risk), ("ALERT","MAXIMUM CAPACITY EXCEEDED")]
  
    else:
        info = [("Status", message), ("Current capacity", cur_inside), ("Risk level", risk)]
    
    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # check to see if we should write the frame to disk
    if writer is not None:
        writer.write(frame)
        
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # increment the total number of frames processed thus far and then update the FPS counter 
    totalFrames += 1
    fps.update()
    
# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# check to see if we need to release the video writer pointer
if writer is not None:
    writer.release()

# release the video file pointer
vs.release()

# close any open windows
cv2.destroyAllWindows()
