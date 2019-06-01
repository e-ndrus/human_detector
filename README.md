# human_detector

Human detector using Tensorflow and pretrained SSD MobileNet v1 (COCO). The script detects only human objects, waits for a specified period and then emits a sound alarm and/or sends an email with a picture of a detected person.

For alarm to work on Linux, you'll need sudo apt install sox
You'll also need to set up your app password in your Google account if you would like to use Gmail. More on this here: https://support.google.com/accounts/answer/185833?hl=en



Code adapted from Tensorflow Object Detection Framework
https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
Tensorflow Object Detection Detector
and https://gist.github.com/madhawav/1546a4b99c8313f06c0b2d7d7b4a09e2#file-tensorflow-human-detection-py
and https://pythonprogramming.net/introduction-use-tensorflow-object-detection-api-tutorial/

Sending emails adapted from https://github.com/samlopezf/Python-Email/blob/master/send_email.py