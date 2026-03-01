import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tarfile
import urllib.request
import numpy as np
import tensorflow as tf
tf.gfile = tf.io.gfile
tf.compat.v1.disable_eager_execution()

import cv2
import serial
import time
import win32com.client
from PIL import ImageFont

# -------------------- PILLOW FIX --------------------
if not hasattr(ImageFont.FreeTypeFont, "getsize"):
    def getsize(self, text):
        bbox = self.getbbox(text)
        return (bbox[2] - bbox[0], bbox[3] - bbox[1])
    ImageFont.FreeTypeFont.getsize = getsize

# -------------------- SERIAL (ARDUINO) --------------------
try:
    arduino = serial.Serial('COM15', 9600, timeout=1)  # CHANGE IF NEEDED
    time.sleep(2)
    print("Arduino Connected")
except:
    print("Arduino NOT Connected")
    arduino = None

# -------------------- STABLE WINDOWS TTS --------------------
speaker = win32com.client.Dispatch("SAPI.SpVoice")

def speak(text):
    if text.strip():
        speaker.Speak(text, 1)   # 1 = async (non-blocking)

# Speaking control
last_spoken_distance = None
last_spoken_time = 0
DISTANCE_CHANGE_THRESHOLD = 3
SPEAK_DELAY = 2

# -------------------- MODEL SETUP --------------------
MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

if not os.path.exists(PATH_TO_CKPT):
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
    urllib.request.urlretrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        if 'frozen_inference_graph.pb' in os.path.basename(file.name):
            tar_file.extract(file, os.getcwd())

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

if not os.path.exists(PATH_TO_LABELS):
    os.makedirs(os.path.dirname(PATH_TO_LABELS), exist_ok=True)
    url = "https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt"
    urllib.request.urlretrieve(url, PATH_TO_LABELS)

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# -------------------- LOAD GRAPH --------------------
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        od_graph_def.ParseFromString(fid.read())
        tf.import_graph_def(od_graph_def, name='')

# -------------------- CAMERA --------------------
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

print("System Started. Press Q to Quit.")

with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:

        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # -------- READ DISTANCE --------
            distance = None
            if arduino:
                try:
                    if arduino.in_waiting > 0:
                        line = arduino.readline().decode().strip()
                        if line.isdigit():
                            distance = int(line)
                except:
                    distance = None

            image_np_expanded = np.expand_dims(frame, axis=0)

            boxes_out, scores_out, classes_out, num_detections_out = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded}
            )

            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes_out),
                np.squeeze(classes_out).astype(np.int32),
                np.squeeze(scores_out),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=5
            )

            spoken_objects = []
            for i in range(int(num_detections_out[0])):
                if scores_out[0][i] > 0.6:
                    obj_name = category_index[int(classes_out[0][i])]['name']
                    if obj_name not in spoken_objects:
                        spoken_objects.append(obj_name)

            # -------- SPEAK LOGIC --------
            current_time = time.time()

            if spoken_objects and distance is not None:
                if distance <= 40:

                    distance_changed = (
                        last_spoken_distance is None or
                        abs(distance - last_spoken_distance) >= DISTANCE_CHANGE_THRESHOLD
                    )

                    enough_time_passed = (current_time - last_spoken_time) > SPEAK_DELAY

                    if distance_changed or enough_time_passed:
                        message = f"{', '.join(spoken_objects)} is {distance} centimeters ahead"
                        print(message)
                        speak(message)
                        last_spoken_distance = distance
                        last_spoken_time = current_time

            cv2.imshow("Smart Stick Vision", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# -------------------- CLEAN EXIT --------------------
cap.release()
cv2.destroyAllWindows()

if arduino:
    arduino.close()