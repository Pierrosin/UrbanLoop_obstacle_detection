from multiprocessing.pool import IMapIterator
from re import I
from scipy.special import comb
import cv2
import time
import numpy as np

ROOT_COLAB = './'
YOLO_CONFIG = ROOT_COLAB + 'oc_data/'

COCO_LABELS_FILE = YOLO_CONFIG + 'coco.names'
YOLO_CONFIG_FILE = YOLO_CONFIG + 'yolov4.cfg'
YOLO_WEIGHTS_FILE = YOLO_CONFIG + 'yolov4.weights'

COCO_LABELS_FILE_URBANLOOP = YOLO_CONFIG + 'urbanloop.names'
YOLO_CONFIG_FILE_URBANLOOP = YOLO_CONFIG + 'urbanloop_yolov4.cfg'
YOLO_WEIGHTS_FILE_URBANLOOP = YOLO_CONFIG + 'urbanloop_yolov4_best5.weights'

IMAGE = cv2.imread(ROOT_COLAB + '/UrbanLoop.jpg')

LABELS_FROM_FILE = False
CONFIDENCE_MIN = 0.5

with open(COCO_LABELS_FILE, 'rt') as f:
    labels = f.read().rstrip('\n').split('\n')
with open(COCO_LABELS_FILE_URBANLOOP, 'rt') as f:
    labels+= f.read().rstrip('\n').split('\n')
np.random.seed(45)
BOX_COLORS = np.random.randint(
    0, 255, size=(len(labels), 3), dtype="uint8")
yolo = cv2.dnn.readNetFromDarknet(YOLO_CONFIG_FILE, YOLO_WEIGHTS_FILE)
yololayers = [yolo.getLayerNames()[i[0] - 1]
              for i in yolo.getUnconnectedOutLayers()]
print(cv2.__version__)
yolo_urbanloop = cv2.dnn.readNetFromDarknet(YOLO_CONFIG_FILE_URBANLOOP, YOLO_WEIGHTS_FILE_URBANLOOP)
yololayers_urbanloop = [yolo_urbanloop.getLayerNames()[i[0] - 1]
              for i in yolo_urbanloop.getUnconnectedOutLayers()]


# Little function to resize in keeping the format ratio

def ResizeWithAspectRatio(_image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    image = _image.copy()
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)

# Displays real-time obstacle detection from a camera, a video or a picture

def obstacle_detection(source):
    # load video class
    cap = cv2.VideoCapture(source)

    # value initialize
    left_maxpoint = [0] * 50
    right_maxpoint = [195] * 50

    # convolution filter
    kernel = np.array([
        [-1, 1, 0, 1, -1],
        [-1, 1, 0, 1, -1],
        [-1, 1, 0, 1, -1],
        [-1, 1, 0, 1, -1],
        [-1, 1, 0, 1, -1]
    ])

    # Next frame availability
    r = True
    first = True
    key = 0

    # analyse frame by frame
    while r and key != 27:
        #--------------RAILWAY DETECTION--------------------

        r, frame = cap.read()
        if frame is None:
            break

        time0 = time.time()

        # cut away invalid frame area
        # valid_frame = frame
        valid_frame = ResizeWithAspectRatio(frame, width=1000)
        # original_frame = valid_frame.copy()

        # gray scale transform
        gray_frame = cv2.cvtColor(valid_frame, cv2.COLOR_BGR2GRAY)

        # histogram equalization image
        histeqaul_frame = cv2.equalizeHist(gray_frame)

        # apply gaussian blur
        blur_frame = cv2.GaussianBlur(histeqaul_frame, (5, 5), 5)

        # merge current frame and last frame
        if first is True:
            merge_frame = blur_frame
            first = False
            old_valid_frame = merge_frame.copy()
        else:
            merge_frame = cv2.addWeighted(blur_frame, 0.2, old_valid_frame, 0.8, 0)
            old_valid_frame = merge_frame.copy()

        # convolution filter
        conv_frame = cv2.filter2D(merge_frame, -1, kernel)

        # initialization for sliding window property
        #sliding_window = [700, 900, 1000, 1200]
        #slide_interval = 20
        #slide_height = 20
        #slide_width = 50

        sliding_window = [350, 450, 550, 650]
        slide_interval = 20
        slide_height = 20
        slide_width = 50

        # initialization for bezier curve variables
        left_points = []
        right_points = []

        # define count value
        count = 0
        for i in range(550, 150, -slide_interval):
            # get edges in sliding window
            left_edge = conv_frame[i:i + slide_height, sliding_window[0]:sliding_window[1]].sum(axis=0)
            right_edge = conv_frame[i:i + slide_height, sliding_window[2]:sliding_window[3]].sum(axis=0)

            # left railroad line processing
            if left_edge.argmax() > 0:
                left_maxindex = sliding_window[0] + left_edge.argmax()
                left_maxpoint[count] = left_maxindex
                cv2.line(valid_frame, (left_maxindex, i + int(slide_height / 2)),
                         (left_maxindex, i + int(slide_height / 2)), (255, 255, 255), 5, cv2.LINE_AA)
                left_points.append([left_maxindex, i + int(slide_height / 2)])
                sliding_window[0] = max(330, left_maxindex - int(slide_width / 4 + (slide_width + 10) / (count + 1)))
                sliding_window[1] = min(670, left_maxindex + int(slide_width / 4 + (slide_width + 10) / (count + 1)))
                #cv2.rectangle(valid_frame, (sliding_window[0], i + slide_height), (sliding_window[1], i), (255, 0, 0), 1)

            # right railroad line processing
            if right_edge.argmax() > 0:
                right_maxindex = sliding_window[2] + right_edge.argmax()
                right_maxpoint[count] = right_maxindex
                cv2.line(valid_frame, (right_maxindex, i + int(slide_height / 2)),
                         (right_maxindex, i + int(slide_height / 2)), (255, 255, 255), 5, cv2.LINE_AA)
                right_points.append([right_maxindex, i + int(slide_height / 2)])
                sliding_window[2] = max(330, right_maxindex - int(slide_width / 4 + (slide_width + 10) / (count + 1)))
                sliding_window[3] = min(670, right_maxindex + int(slide_width / 4 + (slide_width + 10) / (count + 1)))
                #cv2.rectangle(valid_frame, (sliding_window[2], i + slide_height), (sliding_window[3], i), (255, 0, 0), 1)
            count += 1

        # bezier curve process
        bezier_left_xval, bezier_left_yval = bezier_curve(left_points, 50)
        bezier_right_xval, bezier_right_yval = bezier_curve(right_points, 50)

        bezier_left_points = []
        bezier_right_points = []
        try:
            old_point = (bezier_left_xval[0], bezier_left_yval[0])
            for point in zip(bezier_left_xval, bezier_left_yval):
                cv2.line(valid_frame, old_point, point, (255, 0, 0), 2, cv2.LINE_AA)
                old_point = point
                bezier_left_points.append(point)

            old_point = (bezier_right_xval[0], bezier_right_yval[0])
            for point in zip(bezier_right_xval, bezier_right_yval):
                cv2.line(valid_frame, old_point, point, (255, 0, 0), 2, cv2.LINE_AA)
                old_point = point
                bezier_right_points.append(point)
        except IndexError:
            pass

        #--------------OBJECT DETECTION----------------------
    
        blobimage = cv2.dnn.blobFromImage(valid_frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        yolo.setInput(blobimage)
        layerOutputs = yolo.forward(yololayers)
    
        blobimage_urbanloop = cv2.dnn.blobFromImage(valid_frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        yolo_urbanloop.setInput(blobimage_urbanloop)
        layerOutputs_urbanloop = yolo_urbanloop.forward(yololayers_urbanloop)

        boxes_detected = []
        confidences_scores = []
        labels_detected = []
        (h, w) = valid_frame.shape[:2]
        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # Take only predictions with confidence more than CONFIDENCE_MIN thresold
                if confidence > CONFIDENCE_MIN:
                    # Bounding box
                    box = detection[0:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")

                    # Use the center (x, y)-coordinates to derive the top and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our result list (detection)
                    boxes_detected.append([x, y, int(width), int(height)])
                    confidences_scores.append(float(confidence))
                    labels_detected.append(classID)

        # loop over each of the layer outputs
        for output in layerOutputs_urbanloop:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # Take only predictions with confidence more than CONFIDENCE_MIN thresold
                if confidence > CONFIDENCE_MIN:
                    # Bounding box
                    box = detection[0:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")

                    # Use the center (x, y)-coordinates to derive the top and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our result list (detection)
                    boxes_detected.append([x, y, int(width), int(height)])
                    confidences_scores.append(float(confidence))
                    labels_detected.append(classID+80)

        final_boxes = cv2.dnn.NMSBoxes(
            boxes_detected, confidences_scores, 0.5, 0.5)

        # loop through the final set of detections remaining after NMS and draw bounding box and write text
        for max_valueid in final_boxes:
            max_class_id = max_valueid[0]

            # extract the bounding box coordinates
            (x, y) = (boxes_detected[max_class_id]
                    [0], boxes_detected[max_class_id][1])
            (w, h) = (boxes_detected[max_class_id]
                    [2], boxes_detected[max_class_id][3])

            # locate the object relative to the railway
            left_point_index = 0
            while left_point_index < len(bezier_left_points) and (bezier_left_points[left_point_index][1] < y + h):
                left_point_index += 1

            if left_point_index == len(bezier_left_points):
                left_point_index = len(bezier_left_points) -1

            right_point_index = 0
            while right_point_index < len(bezier_right_points) and (bezier_right_points[right_point_index][1] < y + h):
                right_point_index += 1

            if right_point_index == len(bezier_right_points):
                right_point_index = len(bezier_right_points) -1

            x_left_railway, x_right_railway = bezier_left_points[left_point_index][0], bezier_right_points[right_point_index][0]
            
            # find if the object is on the railway and is not a UrbanLoop vehicule
            if (x + w >= x_left_railway) and (x <= x_right_railway) and labels[labels_detected[max_class_id]] != 'Capsule UrbanLoop':
                # DANGER : the object is on the railway

                # draw a DANGER bounding box rectangle and label on the image
                color = [0, 0, 255]
                cv2.rectangle(valid_frame, (x, y), (x + w, y + h), color, 1)

                score = str(
                    round(float(confidences_scores[max_class_id]) * 100, 1)) + "%"
                text = "DANGER: {}: {}".format(labels[labels_detected[max_class_id]], score)
                cv2.putText(valid_frame, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            else:
                # SAFE : the object is not on the railway

                # draw a SAFE bounding box rectangle and label on the image
                color = [0, 255, 0]
                cv2.rectangle(valid_frame, (x, y), (x + w, y + h), color, 1)

                score = str(
                    round(float(confidences_scores[max_class_id]) * 100, 1)) + "%"
                text = "SAFE: {}: {}".format(labels[labels_detected[max_class_id]], score)
                cv2.putText(valid_frame, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        time1 = time.time()

        print('Time analyse image :', time1-time0)

        cv2.imshow('Obstacle detection', valid_frame)
        key = cv2.waitKey(1)
    
    while key != 27:
        key = cv2.waitKey(100)
    cv2.destroyAllWindows()

    print('finish')


def bezier_curve(points, ntimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ..[Xn, Yn] ]
        ntimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    def bernstein_poly(i, n, t):
        """
         The Bernstein polynomial of n, i as a function of t
        """
        return comb(n, i) * (t ** (n - i)) * (1 - t) ** i

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, ntimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)
    return xvals.astype('int32'), yvals.astype('int32')


# Generates obstacle detection video from a video

def generate_video_detection(video_name):
    cap = cv2.VideoCapture(video_name)
    success, image = cap.read()
    height,width,layers=ResizeWithAspectRatio(image, width=1000).shape

    video=cv2.VideoWriter('./'+video_name[:-4]+'Analysed.avi',0,25,(width,height))

    # value initialize
    left_maxpoint = [0] * 50
    right_maxpoint = [195] * 50

    # convolution filter
    kernel = np.array([
        [-1, 1, 0, 1, -1],
        [-1, 1, 0, 1, -1],
        [-1, 1, 0, 1, -1],
        [-1, 1, 0, 1, -1],
        [-1, 1, 0, 1, -1]
    ])

    # Next frame availability
    r = True
    first = True
    key = 0

    # analyse frame by frame
    while r and key != 27:
        #--------------RAILWAY DETECTION--------------------

        r, frame = cap.read()
        if frame is None:
            break

        time0 = time.time()

        # cut away invalid frame area
        # valid_frame = frame
        valid_frame = ResizeWithAspectRatio(frame, width=1000)
        # original_frame = valid_frame.copy()

        # gray scale transform
        gray_frame = cv2.cvtColor(valid_frame, cv2.COLOR_BGR2GRAY)

        # histogram equalization image
        histeqaul_frame = cv2.equalizeHist(gray_frame)

        # apply gaussian blur
        blur_frame = cv2.GaussianBlur(histeqaul_frame, (5, 5), 5)

        # merge current frame and last frame
        if first is True:
            merge_frame = blur_frame
            first = False
            old_valid_frame = merge_frame.copy()
        else:
            merge_frame = cv2.addWeighted(blur_frame, 0.2, old_valid_frame, 0.8, 0)
            old_valid_frame = merge_frame.copy()

        # convolution filter
        conv_frame = cv2.filter2D(merge_frame, -1, kernel)

        # initialization for sliding window property
        #sliding_window = [700, 900, 1000, 1200]
        #slide_interval = 20
        #slide_height = 20
        #slide_width = 50

        sliding_window = [350, 450, 550, 650]
        slide_interval = 20
        slide_height = 20
        slide_width = 50

        # initialization for bezier curve variables
        left_points = []
        right_points = []

        # define count value
        count = 0
        for i in range(550, 150, -slide_interval):
            # get edges in sliding window
            left_edge = conv_frame[i:i + slide_height, sliding_window[0]:sliding_window[1]].sum(axis=0)
            right_edge = conv_frame[i:i + slide_height, sliding_window[2]:sliding_window[3]].sum(axis=0)

            # left railroad line processing
            if left_edge.argmax() > 0:
                left_maxindex = sliding_window[0] + left_edge.argmax()
                left_maxpoint[count] = left_maxindex
                cv2.line(valid_frame, (left_maxindex, i + int(slide_height / 2)),
                         (left_maxindex, i + int(slide_height / 2)), (255, 255, 255), 5, cv2.LINE_AA)
                left_points.append([left_maxindex, i + int(slide_height / 2)])
                sliding_window[0] = max(330, left_maxindex - int(slide_width / 4 + (slide_width + 10) / (count + 1)))
                sliding_window[1] = min(670, left_maxindex + int(slide_width / 4 + (slide_width + 10) / (count + 1)))
                #cv2.rectangle(valid_frame, (sliding_window[0], i + slide_height), (sliding_window[1], i), (255, 0, 0), 1)

            # right railroad line processing
            if right_edge.argmax() > 0:
                right_maxindex = sliding_window[2] + right_edge.argmax()
                right_maxpoint[count] = right_maxindex
                cv2.line(valid_frame, (right_maxindex, i + int(slide_height / 2)),
                         (right_maxindex, i + int(slide_height / 2)), (255, 255, 255), 5, cv2.LINE_AA)
                right_points.append([right_maxindex, i + int(slide_height / 2)])
                sliding_window[2] = max(330, right_maxindex - int(slide_width / 4 + (slide_width + 10) / (count + 1)))
                sliding_window[3] = min(670, right_maxindex + int(slide_width / 4 + (slide_width + 10) / (count + 1)))
                #cv2.rectangle(valid_frame, (sliding_window[2], i + slide_height), (sliding_window[3], i), (255, 0, 0), 1)
            count += 1

        # bezier curve process
        bezier_left_xval, bezier_left_yval = bezier_curve(left_points, 50)
        bezier_right_xval, bezier_right_yval = bezier_curve(right_points, 50)

        bezier_left_points = []
        bezier_right_points = []
        try:
            old_point = (bezier_left_xval[0], bezier_left_yval[0])
            for point in zip(bezier_left_xval, bezier_left_yval):
                cv2.line(valid_frame, old_point, point, (255, 0, 0), 2, cv2.LINE_AA)
                old_point = point
                bezier_left_points.append(point)

            old_point = (bezier_right_xval[0], bezier_right_yval[0])
            for point in zip(bezier_right_xval, bezier_right_yval):
                cv2.line(valid_frame, old_point, point, (255, 0, 0), 2, cv2.LINE_AA)
                old_point = point
                bezier_right_points.append(point)
        except IndexError:
            pass

        #--------------OBJECT DETECTION----------------------
    
        blobimage = cv2.dnn.blobFromImage(valid_frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        yolo.setInput(blobimage)
        layerOutputs = yolo.forward(yololayers)
    
        blobimage_urbanloop = cv2.dnn.blobFromImage(valid_frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        yolo_urbanloop.setInput(blobimage_urbanloop)
        layerOutputs_urbanloop = yolo_urbanloop.forward(yololayers_urbanloop)

        boxes_detected = []
        confidences_scores = []
        labels_detected = []
        (h, w) = valid_frame.shape[:2]
        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # Take only predictions with confidence more than CONFIDENCE_MIN thresold
                if confidence > CONFIDENCE_MIN:
                    # Bounding box
                    box = detection[0:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")

                    # Use the center (x, y)-coordinates to derive the top and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our result list (detection)
                    boxes_detected.append([x, y, int(width), int(height)])
                    confidences_scores.append(float(confidence))
                    labels_detected.append(classID)

        # loop over each of the layer outputs
        for output in layerOutputs_urbanloop:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # Take only predictions with confidence more than CONFIDENCE_MIN thresold
                if confidence > CONFIDENCE_MIN:
                    # Bounding box
                    box = detection[0:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")

                    # Use the center (x, y)-coordinates to derive the top and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our result list (detection)
                    boxes_detected.append([x, y, int(width), int(height)])
                    confidences_scores.append(float(confidence))
                    labels_detected.append(classID+80)

        final_boxes = cv2.dnn.NMSBoxes(
            boxes_detected, confidences_scores, 0.5, 0.5)

        # loop through the final set of detections remaining after NMS and draw bounding box and write text
        for max_valueid in final_boxes:
            max_class_id = max_valueid[0]

            # extract the bounding box coordinates
            (x, y) = (boxes_detected[max_class_id]
                    [0], boxes_detected[max_class_id][1])
            (w, h) = (boxes_detected[max_class_id]
                    [2], boxes_detected[max_class_id][3])

            # locate the object relative to the railway
            left_point_index = 0
            while left_point_index < len(bezier_left_points) and (bezier_left_points[left_point_index][1] < y + h):
                left_point_index += 1

            if left_point_index == len(bezier_left_points):
                left_point_index = len(bezier_left_points) -1

            right_point_index = 0
            while right_point_index < len(bezier_right_points) and (bezier_right_points[right_point_index][1] < y + h):
                right_point_index += 1

            if right_point_index == len(bezier_right_points):
                right_point_index = len(bezier_right_points) -1

            x_left_railway, x_right_railway = bezier_left_points[left_point_index][0], bezier_right_points[right_point_index][0]

            # find if the object is on the railway and is not a UrbanLoop vehicule
            if (x + w >= x_left_railway) and (x <= x_right_railway) and labels[labels_detected[max_class_id]] != 'Capsule UrbanLoop':
                # DANGER : the object is on the railway

                # draw a DANGER bounding box rectangle and label on the image
                color = [0, 0, 255]
                cv2.rectangle(valid_frame, (x, y), (x + w, y + h), color, 1)

                score = str(
                    round(float(confidences_scores[max_class_id]) * 100, 1)) + "%"
                text = "DANGER: {}: {}".format(labels[labels_detected[max_class_id]], score)
                cv2.putText(valid_frame, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            else:
                # SAFE : the object is not on the railway

                # draw a SAFE bounding box rectangle and label on the image
                color = [0, 255, 0]
                cv2.rectangle(valid_frame, (x, y), (x + w, y + h), color, 1)

                score = str(
                    round(float(confidences_scores[max_class_id]) * 100, 1)) + "%"
                text = "SAFE: {}: {}".format(labels[labels_detected[max_class_id]], score)
                cv2.putText(valid_frame, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        time1 = time.time()

        print('Time analyse image :', time1-time0)

        video.write(valid_frame)
        
        key = cv2.waitKey(1)

    cv2.destroyAllWindows()

    video.release()

    print('finish')
        
    
#generate_video_detection("DémonstrationDétectionObstacleUrbanLoop.mp4")

#obstacle_detection("Vidéos/DémonstrationDétectionObstacleUrbanLoop.mp4")
obstacle_detection("Photos/Photos non analysées/SituationUrbanLoopSafe.png")
#obstacle_detection(0)