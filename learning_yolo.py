from ultralytics import YOLO

model = YOLO('yolov8m.yaml')  

results = model.train(data='hack.yaml', epochs=100, imgsz=640, degrees = 90, fliplr = 0.5)

print(model.val())