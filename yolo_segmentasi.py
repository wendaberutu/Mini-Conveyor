from ultralytics import YOLO

model = YOLO(r"D:\wenda\Mini Conveyor\runs\segment\model_segmentation_custom\weights\best.pt")
print("Nama class:", model.names)
print("Jumlah class:", len(model.names))
print("Info model:")
model.info()

