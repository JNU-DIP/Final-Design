from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load an official model

# Predict with the model
results = model("https://ultralytics.com/images/bus.jpg", save=True)  # predict on an image
