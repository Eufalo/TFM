# USAGE
# python3 Person_Detection_edge.py --model mobilenet_ssd_v2/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite --labels mobilenet_ssd_v2/coco_labels.txt --input pedestrians.mp4 


# import the necessary packages
from edgetpu.detection.engine import DetectionEngine
from imutils.video import VideoStream
from PIL import Image
import numpy as np
import argparse
import imutils
import time
import yaml
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
        help="path to (optional) input video file")
ap.add_argument("-m", "--model", required=True,
    help="path to TensorFlow Lite object detection model")
ap.add_argument("-l", "--labels", required=True,
    help="path to labels file")
ap.add_argument("-c", "--confidence", type=float, default=0.6,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the labels dictionary
print("[INFO] parsing class labels...")
labels = {}

# loop over the class labels file
for row in open(args["labels"]):
    # unpack the row and update the labels dictionary
    (classID, label) = row.strip().split(maxsplit=1)
    labels[int(classID)] = label.strip()

# load the Google Coral object detection model
print("[INFO] loading Coral model...")
model = DetectionEngine(args["model"])

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)#VideoStream(src=0).start()
#vs = VideoStream(usePiCamera=False).start()
time.sleep(2.0)
fps = ""
detectfps = ""
framecount = 0
detectframecount = 0
time1 = 0
time2 = 0

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 700 pixels
    t1 = time.perf_counter()
    (grabbed, frame) = vs.read()
    if not grabbed:
        break
    frame = imutils.resize(frame, width=700,height=700)
    orig = frame.copy()
    # prepare the frame for object detection by converting (1) it
    # from BGR to RGB channel ordering and then (2) from a NumPy
    # array to PIL image format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    
    # make predictions on the input frame
    start = time.time()
    results = model.detect_with_image(frame, threshold=args["confidence"],
    keep_aspect_ratio=True, relative_coord=False)

    end = time.time()
    
    detectframecount += 1
    # loop over the results
    for i,r in enumerate(results):
            # extract the bounding box and box and predicted class label      
            box = r.bounding_box.flatten().astype("int")
            (startX, startY, endX, endY) = box
            color = (0, 255, 0)
            #Trasnform the detection to the bird eye
            cX = int((startX+endX)/2)
            cY=int((startY+endY)/2)
            
           
            # draw the bounding box and label on the image
            cv2.rectangle(orig, (startX, startY), (endX, endY),
                        color, 2)
                
            #Centroid circle drow 
            cv2.circle(orig, (cX, cY), 5, color, 1)
            
            
            cv2.putText(orig, fps, (10, orig.shape[0] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
                    

                

    # show the output frame and wait for a key press
    cv2.imshow("Frame", orig)
    
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
            break
    # FPS calculation
    framecount += 1
    if framecount >= 15:
        fps       = "{:.1f} FPS".format(time1/15)
        detectfps = "(Detection) {:.1f} FPS".format(detectframecount/time2)
        framecount = 0
        detectframecount = 0
        time1 = 0
        time2 = 0
    t2 = time.perf_counter()
    elapsedTime = t2-t1
    time1 += 1/elapsedTime
    time2 += elapsedTime
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

