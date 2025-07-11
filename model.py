from ultralytics import YOLO

# Inisialisasi model dari file arsitektur custom
model = YOLO("yolov8-seg.yaml", task='segment')  

# Training model
model.train(
    data=r"D:\wenda\Mini Conveyor\dataset\data.yaml",  # path ke file data.yaml kamu
    epochs=200,
    imgsz=320,
    batch=8,
    name="yolov8n-walet-segment",
    
)

