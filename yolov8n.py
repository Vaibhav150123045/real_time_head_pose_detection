from ultralytics import YOLO

# Load pretrained YOLOv8 model (choose one)
model = YOLO('yolov8n.pt')  # Nano (fastest)
# model = YOLO('yolov8s.pt')  # Small (balanced)
# model = YOLO('yolov8m.pt')  # Medium (accurate)

# Export to ONNX with dynamic batch size for real-time
model.export(format='onnx', simplify=True, opset=12, dynamic=False, imgsz=640)
