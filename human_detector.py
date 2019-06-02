# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector
# and https://gist.github.com/madhawav/1546a4b99c8313f06c0b2d7d7b4a09e2#file-tensorflow-human-detection-py
# and https://pythonprogramming.net/introduction-use-tensorflow-object-detection-api-tutorial/

import numpy as np
import tensorflow as tf
import cv2
import time
import os
import send_email
import sys
import urllib.request as urllib

class HumanDetector:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        print("Elapsed Time:", end_time-start_time)

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()

    def url_to_image(self, ip_webcam):
        # for this option to work, you need an IP Webcam installed on your phone
        # tested on Android
        # download the image, convert it to a NumPy array, 
        # and then read it into OpenCV format
        resp = urllib.urlopen(f'http://{ip_webcam}/shot.jpg')
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    	
        # return the image
        return image

if __name__ == "__main__":
    model_path = 'ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'
    odapi = HumanDetector(path_to_ckpt=model_path)
    threshold = 0.7
    cap = cv2.VideoCapture(0)
    c=0
    phone_cam = True if (len(sys.argv[0:]) > 1 and sys.argv[1] == 'phone') else False

    while True:
        if phone_cam:
            ip = sys.argv[2] if (len(sys.argv) >= 2) else "192.168.43.20:8080"
            img = odapi.url_to_image(ip)
        else:
            r, img = cap.read()
        
        img = cv2.resize(img, (1280, 720))
        boxes, scores, classes, num = odapi.processFrame(img)

        # Visualization of the results of a detection.
        for i in range(len(boxes)):
            # Class 1 represents human
            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
                print(c, 'Person detected')
                c+=1
        if c==50:
            duration = 2  # seconds
            freq = 440  # Hz
            os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
            cv2.imwrite('human_detected.png', img)
            send_email.sendDetectedImageToEmail()
            c = 0

        cv2.imshow("preview", img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break