import ast
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
import cv2
import torch
import numpy as np
import time
from manualThreading import WebcamStream
from ONNX_Yolov8 import YOLOv8

cap = WebcamStream(0)  # Use the default camera
cap.start()


model_path = "yolov8n.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.3, iou_thres=0.5)


while cap.stopped is not True:
    img = cap.read()
    start_time = time.time()

    # Update object localizer
    boxes, scores, class_ids = yolov8_detector(img)

    combined_img = yolov8_detector.draw_detections(img)

    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(combined_img, f'FPS: {fps:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Detected Objects", combined_img)

    # Press key q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.stop()
cv2.destroyAllWindows()
detector = None  # Clear the detector instance
torch.cuda.empty_cache()  # Clear CUDA memory if using GPU
print("Released all resources and exited cleanly.")
