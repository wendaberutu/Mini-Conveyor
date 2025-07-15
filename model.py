from ultralytics import YOLO

# Inisialisasi model dari file arsitektur custom
model = YOLO(r"D:\wenda\Mini Conveyor\runs\segment\model_segmentation1\weights\last.pt", task='segment')  

# Training model
model.train(
    data=r"D:\wenda\Mini Conveyor\dataset\data.yaml",  
    epochs=200,
    imgsz=320,
    batch=8,
    lrf =0.001,
    lr0=0.01,
    name="model_segmentation_custom", 
    mosaic=True,  
)






