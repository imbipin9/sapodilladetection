import cv2
from pytorchyolo import detect, models

classes = ["immaturesapodilla", "maturesapodilla", "nonsapodilla"]
detect.detect_directory(
          "./yolov3-custom.cfg",
          "./yolov3_ckpt_997.pth", 
          "./dataset/v2/images",
          classes,
          "./outputs", nms_thres = 0.2, conf_thres=0.2)

