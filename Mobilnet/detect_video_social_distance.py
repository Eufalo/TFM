# USAGE
# python detect_video.py --model mobilenet_ssd_v2/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite --labels mobilenet_ssd_v2/coco_labels.txt

# import the necessary packages
from edgetpu.detection.engine import DetectionEngine
from scipy.spatial import distance as dist
import birdView_tranform as bd_vie_trans
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

#Configure the bird eye
with open("config_bird_view.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile)
width_og, height_og = 0,0
corner_points = []
for section in cfg:
    corner_points.append(cfg["image_parameters"]["p1"])
    corner_points.append(cfg["image_parameters"]["p2"])
    corner_points.append(cfg["image_parameters"]["p3"])
    corner_points.append(cfg["image_parameters"]["p4"])
    width_og = int(cfg["image_parameters"]["width_og"])
    height_og = int(cfg["image_parameters"]["height_og"])
    img_path = cfg["image_parameters"]["img_path"]
    #size_frame = cfg["image_parameters"]["size_frame"]

matrix,imgOutput = bd_vie_trans.compute_perspective_transform(corner_points,width_og,height_og,cv2.imread(img_path))
height,width,_ = imgOutput.shape
blank_image = np.zeros((height,width,3), np.uint8)
height = blank_image.shape[0]
width = blank_image.shape[1]

dim = (700, 400)
# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 700 pixels
    #frame = vs.read()
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
    
    img = cv2.imread("bird_eye_img.png")
    bird_view_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    # make predictions on the input frame
    start = time.time()
    results = model.detect_with_image(frame, threshold=args["confidence"],
    keep_aspect_ratio=True, relative_coord=False)

    end = time.time()
    
    # initialize the set of indexes that violate the minimum social distance
    violate = set()
    
    #Return only the person detection 
    rects_ = np.array([r.bounding_box.flatten().astype("int") if r.label_id==0 else "" for r in results])
    centroids=bd_vie_trans.rect_to_centroids(rects_)
    if len(centroids) >= 2:
           
        D = dist.cdist(centroids, centroids, metric="euclidean")
        
        # loop over the upper triangular of the distance matrix
        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                # check to see if the distance between any two
                # centroid pairs is less than the configured number
                # of pixels
                if D[i, j] < 50:
                    # update our violation set with the indexes of
                    # the centroid pairs
                    violate.add(i)
                    violate.add(j)

   
    # loop over the results
    for i,r in enumerate(results):
            # extract the bounding box and box and predicted class label      
            box = r.bounding_box.flatten().astype("int")
            (startX, startY, endX, endY) = box
            color = (0, 255, 0)
            if i in violate:
                color =(0,0,255)
            #Trasnform the detection to the bird eye
            cX = int((startX+endX)/2)
            cY=int((startY+endY)/2)
            
            matrix=bd_vie_trans.compute_perspective_wh_transform(400,400,frame)
            bird_points=bd_vie_trans.compute_point_perspective_transformation(matrix,[cX,endY])
            
            if (r.label_id==0):  
                label = labels[r.label_id]
                #drw bird points 
                x,y = bird_points[0]
                cv2.circle(bird_view_img, (x,y), 25, color, 2)
                cv2.circle(bird_view_img, (x,y), 3, color, -1)
                '''
                # draw the bounding box and label on the image
                cv2.rectangle(orig, (startX, startY), (endX, endY),
                        color, 2)
                '''
                #Centroid circle drow 
                cv2.circle(orig, (cX, cY), 5, color, 1)
                #Drwa social distance
                cv2.ellipse(orig,(cX,endY),(45,20),0,0,360,color,1)

                # draw the total number of social distancing violations on the
                # output frame
                text = "Social Distancing Violations: {}".format(len(violate))
                cv2.putText(orig, text, (10, orig.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
                '''
                y = startY - 15 if startY - 15 > 15 else startY + 15
                text = "{}: {:.2f}%".format(label, r.score * 100)
                cv2.putText(orig, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                '''

    # show the output frame and wait for a key press
    cv2.imshow("Frame", orig)
    cv2.imshow("bird", bird_view_img)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
            break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
