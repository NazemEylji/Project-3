import locale
locale.getpreferredencoding = lambda: "UTF-8"
!pip install ultralytics
import cv2
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt

drive.mount('/content/drive')

data_directory = '/content/drive/MyDrive/Project 3 Data'
test_images_directory = os.path.join(data_directory, 'test_images')
runs_directory = '/content/drive/MyDrive/runs'
model_path = '/content/drive/MyDrive/Project 3 Data/runs/pcb_component_detection/weights/best.pt'

if not os.path.exists(test_images_directory):
    os.makedirs(test_images_directory)
    print(f"Please upload your test images to the folder: {test_images_directory}")
    exit()

model = YOLO(model_path)

predictions_directory = os.path.join(runs_directory, 'predictions_refined')
os.makedirs(predictions_directory, exist_ok=True)

for image_name in os.listdir(test_images_directory):
    image_path = os.path.join(test_images_directory, image_name)
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        print(f"Processing image: {image_name}")

        results = model.predict(
            source=image_path,
            save=True,
            conf=0.25,
            iou=0.25,
            project=predictions_directory,
            name=image_name.split('.')[0]
        )

        print(f"Results saved for {image_name} in {predictions_directory}")

print(f"All predictions completed. Check results in {predictions_directory}")
