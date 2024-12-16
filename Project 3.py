import locale
locale.getpreferredencoding = lambda: "UTF-8"
!pip install ultralytics
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from ultralytics import YOLO

from google.colab import drive
drive.mount('/content/drive')

data_path = '/content/drive/MyDrive/Project 3 Data'
image_path = os.path.join(data_path, 'motherboard_image.JPEG')

image = cv2.imread(image_path)
plt.figure(figsize=(8, 8))
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(8, 8))
plt.title("Grayscale Image")
plt.imshow(gray, cmap='gray')
plt.axis('off')
plt.show()

_, binary_thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
plt.figure(figsize=(8, 8))
plt.title("Binary Threshold")
plt.imshow(binary_thresh, cmap='gray')
plt.axis('off')
plt.show()

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
cleaned_mask = cv2.morphologyEx(binary_thresh, cv2.MORPH_CLOSE, kernel)
plt.figure(figsize=(8, 8))
plt.title("Morphologically Refined Mask")
plt.imshow(cleaned_mask, cmap='gray')
plt.axis('off')
plt.show()

contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv2.contourArea)

tight_mask = np.zeros_like(gray)
cv2.drawContours(tight_mask, [largest_contour], -1, (255), thickness=cv2.FILLED)
plt.figure(figsize=(8, 8))
plt.title("Tight Mask for Motherboard")
plt.imshow(tight_mask, cmap='gray')
plt.axis('off')
plt.show()

final_extracted = cv2.bitwise_and(image, image, mask=tight_mask)

black_background = np.zeros_like(image)
final_output = np.where(tight_mask[..., None] == 255, final_extracted, black_background)
plt.figure(figsize=(8, 8))
plt.title("Final Extracted Motherboard with Black Background")
plt.imshow(cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

output_path = os.path.join(data_path, 'final_extracted_motherboard.JPEG')
cv2.imwrite(output_path, final_output)
print(f"Final output saved at: {output_path}")

model = YOLO('yolov8n.pt')

model.train(
    data=os.path.join(data_path, 'data.yaml'),
    epochs=120,
    imgsz=1024,
    batch=8,
    device=0,
    optimizer='AdamW',
    lr0=0.001,
    weight_decay=0.0005,
    iou=0.25,
    conf=0.25,
    name='pcb_component_detection_refined_v2',
    project=runs_directory
)

print("Training complete. Results saved in:", os.path.join(runs_directory, 'pcb_component_detection'))
