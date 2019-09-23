# Tensorflow Face Tracker and Detector
A mobilenet SSD(single shot multibox detector) based face detector with pretrained model provided, powered by tensorflow [object detection api](https://github.com/tensorflow/models/tree/master/research/object_detection), trained by [WIDERFACE dataset](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/).
Based on such detector and Kalman Filter we have a Face Tacker, which can track the faces in the video, count numbers of faces occur in video, and crop face images from video.

## Features
Speed, run 60fps on a nvidia GTX1080 GPU.

## Dependencies
Tensorflow >= 1.10

Tensorflow object detection api (Please follow the official installation instruction)

OpenCV python

## Usage

### Prepare pre-trained model
I already put a trained detection model of pb format under the model folder, you can directly use it. Or you can also use your own model by replacing mine, you can also choose the format of model only after you modify the code of calling detecton model in start.py

### Prepare video
Put your test video (mp4 format) under the media folder, rename it as test.mp4.

### Run video face track
At the source root
```bash
python start.py
```
After finished the processing, find the output video at media folder.

### Run video detection
At the source root
```bash
python inference_video_face.py
```
After finished the processing, find the output video at media folder.


### Run detection from usb camera

You can see how this face detection works with your web camera.
```
usage:inference_usbCam_face.py (cameraID | filename)
```

Here is an example to use usb camera with cameraID=0.

```bash
python inference_usbCam_face.py 0
```
Note: this running does not save video.



### About Issue

if your output video is blank. A brief reminder is: check the input codec and check the input/output resolution.


## License
Usage of the code and model by yeephycho is under the license of Apache 2.0.

The code is based on GOOGLE tensorflow object detection api. Please refer to the license of tensorflow.

Dataset is based on WIDERFACE dataset. Please refer to the license to the WIDERFACE license.
