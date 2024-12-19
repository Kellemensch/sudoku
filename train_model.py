from ultralytics import YOLO
import torch

# Grid recognition
def train_grid():
    #if you are using gpu else comment this line
    torch.cuda.set_device(0)
    # Load a model
    model = YOLO("yolov8l.pt")  # load a base model (recommended for training)
    model.train(data="/home/baptiste/Documents/python/sudoku_tp/datasets/grid//data.yaml", epochs=100,imgsz=416)  # train the model

def train_boxes():
    #torch.cuda.set_device(0)
    model = YOLO("yolov8l.pt")
    model.train(data="/home/baptiste/Documents/python/sudoku_tp/datasets/boxes/data.yaml", epochs=1,imgsz=416)  # train the model


# Digit rcognition
def train_digits():
    # Load a model
    model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data="mnist", epochs=100, imgsz=32)

train_boxes()