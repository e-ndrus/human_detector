# human_detector

Human detector using Tensorflow and pretrained SSD Inception v2 (COCO). The script detects only human objects, waits for a specified period and then emits a sound alarm (or sends a signal).

Code adapted from Tensorflow Object Detection Framework
https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
Tensorflow Object Detection Detector
and https://gist.github.com/madhawav/1546a4b99c8313f06c0b2d7d7b4a09e2#file-tensorflow-human-detection-py
and https://pythonprogramming.net/introduction-use-tensorflow-object-detection-api-tutorial/

Sending emails adapted from https://github.com/samlopezf/Python-Email/blob/master/send_email.py