import threading
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
import cv2
import torch
import numpy as np
import time
from manualThreading import WebcamStream

import sys
import glob


class YoloDetector:

    def __init__(self, model_name):
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

    def load_model(self, model_name):
        if model_name:
            # Load custom trained weights
            model = torch.hub.load('ultralytics/yolov5', 'custom',
                                   path=model_name,
                                   force_reload=True)
        else:
            # Load pretrained YOLOv5 Nano (official smallest model)
            model = torch.hub.load('ultralytics/yolov5',
                                   'yolov5n',
                                   force_reload=True)

        return model

    def score_frame(self, frame):
        self.model.to(self.device)
        downscale_factor = 2
        width = int(frame.shape[1] / downscale_factor)
        height = int(frame.shape[0] / downscale_factor)
        resized_frame = cv2.resize(frame, (width, height))
        results = self.model(resized_frame)

        labels, cords = results.xyxyn[0][:, -
                                         1].cpu().numpy(), results.xyxyn[0][:, :-1].cpu().numpy()
        return labels, cords

    def class_to_label(self, x):
        return self.classes[int(x)]

    def plot_boxes(self, results, frame, height, width, conf_thres=0.3):
        labels, cord = results
        detections = []

        n = len(labels)
        if n == 0:
            return frame, detections

        x_shape, y_shape = width, height

        for i in range(n):
            row = cord[i]
            if row[4] >= conf_thres:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(
                    row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
                label = self.class_to_label(labels[i])
                if label == 'person':
                    x_center = x1 + (x2 - x1) / 2
                    y_center = y1 + (y2 - y1) / 2
                    tlwh = np.array(
                        [x1, y1, x2 - x1, y2 - y1], dtype=np.float32)
                    confidence = float(row[4].item())
                    feature = 'person'
                    detections.append(
                        ([x1, y1, x2 - x1, y2 - y1], row[4].item(), 'person'))

        return frame, detections


cap = WebcamStream(0)  # Use the default camera
cap.start()


detector = YoloDetector(model_name=None)  # Load the YOLOv5 model

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


object_tracker = DeepSort(
    max_age=5,
    n_init=2,
    nms_max_overlap=1.0,
    max_cosine_distance=0.3,
    nn_budget=None,
    override_track_class=None,
    embedder='mobilenet',
    half=True,
    bgr=True,
    embedder_gpu=True,
    embedder_model_name=None,
    embedder_wts=None,
    polygon=False,
    today=None
)

# Get number of active threads (excluding main thread)
active_threads = threading.active_count() - 1  # Subtract main thread
print(f"Active threads: {active_threads}")


while cap.stopped is not True:
    img = cap.read()
    start_time = time.time()
    results = detector.score_frame(img)
    img, detections = detector.plot_boxes(
        results, img, img.shape[0], img.shape[1], conf_thres=0.5)
    tracks = object_tracker.update_tracks(detections, frame=img)

    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        bbox = ltrb
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                      (255, 0, 0), 2)
        cv2.putText(img, f'ID: {track_id}', (int(bbox[0]), int(bbox[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(img, f'FPS: {fps:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('Object Tracking', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.stop()
cv2.destroyAllWindows()
detector.model = None  # Release the model resources
detector = None  # Clear the detector instance
torch.cuda.empty_cache()  # Clear CUDA memory if using GPU
print("Released all resources and exited cleanly.")
