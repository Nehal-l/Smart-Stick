import os
# ---- Suppress TensorFlow GPU/CUDA Warnings ----
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide INFO, WARNING, and ERROR messages from TensorFlow
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Comment this if you want to use GPU

import tarfile
import urllib.request
import numpy as np
import tensorflow as tf
tf.gfile = tf.io.gfile
tf.compat.v1.disable_eager_execution()

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

import torch
from torch.autograd import Variable as V
from torchvision import transforms as trn
from torch.nn import functional as F

from PIL import Image, ImageFont
import cv2
import queue
import threading
import pyttsx3

# ---- FIX for Pillow >=10 ----
if not hasattr(ImageFont.FreeTypeFont, "getsize"):
    def getsize(self, text):
        bbox = self.getbbox(text)
        return (bbox[2] - bbox[0], bbox[3] - bbox[1])
    ImageFont.FreeTypeFont.getsize = getsize

# -------------------- Thread-safe TTS --------------------
engine = pyttsx3.init()
tts_queue = queue.Queue()

def tts_worker():
    while True:
        text = tts_queue.get()
        if text is None:  # Stop signal
            break
        engine.say(text)
        engine.runAndWait()
        tts_queue.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

def speak_async_safe(text):
    if text.strip():
        tts_queue.put(text)

# -------- Windows-friendly download function --------
def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename} ...")
        try:
            urllib.request.urlretrieve(url, filename)
            print(f"{filename} download complete!")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
    else:
        print(f"{filename} already exists.")

# -------- Places365 model setup --------
arch = 'resnet18'
model_file = f'whole_{arch}_places365_python36.pth.tar'
weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
download_file(weight_url, model_file)

# -------- Object detection model setup --------
MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

if not os.path.exists(PATH_TO_CKPT):
    print('Downloading the object detection model...')
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
    download_file(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        if 'frozen_inference_graph.pb' in os.path.basename(file.name):
            tar_file.extract(file, os.getcwd())
    print('Object detection model downloaded.')
else:
    print('Object detection model already exists.')

# -------- Label map utils --------
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

if not os.path.exists(PATH_TO_LABELS):
    os.makedirs(os.path.dirname(PATH_TO_LABELS), exist_ok=True)
    print("Downloading COCO label map file...")
    url = "https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt"
    urllib.request.urlretrieve(url, PATH_TO_LABELS)
    print("Label map downloaded successfully!")

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# -------- Video capture --------
cap = cv2.VideoCapture(0)

# -------- TensorFlow detection graph --------
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        od_graph_def.ParseFromString(fid.read())
        tf.import_graph_def(od_graph_def, name='')

# -------- Places365 categories file --------
categories_file = 'categories_places365.txt'
categories_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
download_file(categories_url, categories_file)

# -------- Main loop --------
with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        print("Starting video stream... Press 'q' to quit.")
        while True:
            ret, image_np = cap.read()
            if not ret:
                break

            key = cv2.waitKey(2) & 0xFF

            # ---------- Scene Recognition ----------
            if key == ord('b'):
                cv2.imwrite('opencv.jpg', image_np)
                model = torch.load(model_file, map_location=lambda storage, loc: storage)
                model.eval()

                centre_crop = trn.Compose([
                    trn.Resize((256, 256)),
                    trn.CenterCrop(224),
                    trn.ToTensor(),
                    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

                classes = []
                with open(categories_file) as f:
                    for line in f:
                        classes.append(line.strip().split(' ')[0][3:])
                classes = tuple(classes)

                img = Image.open('opencv.jpg')
                input_img = V(centre_crop(img).unsqueeze(0))
                logit = model.forward(input_img)
                h_x = F.softmax(logit, 1).data.squeeze()
                probs, idx = h_x.sort(0, True)

                print('POSSIBLE SCENES ARE:')
                speak_async_safe("Possible scene may be")
                for i in range(5):
                    print(classes[idx[i]])
                    speak_async_safe(classes[idx[i]])

            # ---------- OCR ----------
            if key == ord('r'):
                text = pytesseract.image_to_string(image_np)
                print("Detected text:", text)
                speak_async_safe(text if text.strip() else "No text detected")

            # ---------- Object Detection ----------
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes_tf = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            (boxes, scores, classes_tf, num_detections) = sess.run(
                [boxes, scores, classes_tf, num_detections],
                feed_dict={image_tensor: image_np_expanded}
            )

            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes_tf).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8
            )

            # Speak detected objects (thread-safe)
            spoken_objects = []
            for i in range(int(num_detections[0])):
                if scores[0][i] > 0.5:
                    obj_name = category_index[int(classes_tf[0][i])]['name']
                    if obj_name not in spoken_objects:
                        spoken_objects.append(obj_name)

            if spoken_objects:
                text_to_speak = ", ".join(spoken_objects)
                print(f"Detected: {text_to_speak}")
                speak_async_safe(text_to_speak)

            # Show image
            cv2.imshow('image', cv2.resize(image_np, (1024, 768)))

            # Quit
            if key == ord('q') or key == ord('Q'):
                print("Exiting...")
                cap.release()
                cv2.destroyAllWindows()
                tts_queue.put(None)  # Stop TTS thread
                break