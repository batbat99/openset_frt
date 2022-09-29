import os
import sys
sys.path.insert(1, 'yolo_deepsort')
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils_edited as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

print('modules imported')



framework = tf
weights = 'yolo_deepsort/checkpoints/yolov4-tiny-416'
size = 416
tiny = True
model = 'yolov4'
video = "0"
output = None
output_format = 'XVID'
iou = 0.45
score = 0.5
dont_show = False
info = False
count = False

def init_vars(**kwargs):
    for arg in kwargs:
        globals()[arg] = kwargs[arg]

print('vars init')
    

# Definition of the parameters
max_cosine_distance = 0.4
nn_budget = None
nms_max_overlap = 1.0
    
# initialize deep sort
model_filename = 'yolo_deepsort/model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
# calculate cosine distance metric
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
# initialize tracker
tracker = Tracker(metric)

print('deepsort initialized')

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(tiny, model)
input_size = size
video_path = video



if framework == 'tflite':
    interpreter = tf.lite.Interpreter(model_path=weights)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details)
    print(output_details)
# otherwise load standard tensorflow saved model
else:
    saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

# begin video capture
try:
    vid = cv2.VideoCapture(int(video_path))
except:
    vid = cv2.VideoCapture(video_path)

out = None

    #initialize color map
cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

def read_frame(frame_num = 0):
    
    return_value, frame = vid.read()
    if return_value:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
    else:
        print('Video has ended or failed, try a different video format!')
    frame_num +=1
    print('Frame #: ', frame_num)
    frame_size = frame.shape[:2]
    image_data = cv2.resize(frame, (input_size, input_size))
    image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    start_time = time.time()
    return frame, start_time, image_data, frame_num
    
def detection(frame, image_data):
    # run detections on tflite if flag is set
    if framework == 'tflite':
        interpreter.set_tensor(input_details[0]['index'], image_data)
        interpreter.invoke()
        pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
        # run detections using yolov3 if flag is set
        if model == 'yolov3' and tiny == True:
            boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
    else:
        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=iou,
            score_threshold=score
        )

        # convert data to numpy arrays and slice out unused elements
    num_objects = valid_detections.numpy()[0]
    bboxes = boxes.numpy()[0]
    bboxes = bboxes[0:int(num_objects)]
    scores = scores.numpy()[0]
    scores = scores[0:int(num_objects)]
    classes = classes.numpy()[0]
    classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
    original_h, original_w, _ = frame.shape
    bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
    pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        #allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
    allowed_classes = ['person','bicycle','car','motorbike','bus','truck']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
    names = []
    deleted_indx = []
    for i in range(num_objects):
        class_indx = int(classes[i])
        class_name = class_names[class_indx]
        if class_name not in allowed_classes:
            deleted_indx.append(i)
        else:
            names.append(class_name)
    names = np.array(names)
    count = len(names)
    if count:
        cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
        print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
    bboxes = np.delete(bboxes, deleted_indx, axis=0)
    scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
    features = encoder(frame, bboxes)
    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]


        # run non-maxima supression
    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]
    
        # Call the tracker
    tracker.predict()
    tracker.update(detections)
    
    
    return tracker
def out_video(frame, start_time):
        # calculate frames per second of running detections
    if (time.time() - start_time) != 0 :
        fps = 1.0 / (time.time() - start_time)
    else: fps = 0
    print("FPS: %.2f" % fps)
    result = np.asarray(frame)
    result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
    if not dont_show:
        cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
    if output:
        out.write(result)
    if cv2.waitKey(1) & 0xFF == ord('q'): cv2.destroyAllWindows()