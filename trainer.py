import cv2
import os
import numpy as np
from PIL import Image

# Path where your dataset is stored
dataset_path = "dataset"
trainer_path = "trainer"

# Create trainer folder if not exists
if not os.path.exists(trainer_path):
    os.makedirs(trainer_path)

# LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Haar Cascade for detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Function to load images and labels
def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []

    for image_path in image_paths:
        # Load image and convert to grayscale
        img = Image.open(image_path).convert("L")
        img_np = np.array(img, 'uint8')

        # Extract ID from filename (User.ID.Count.jpg)
        id = int(os.path.split(image_path)[-1].split('.')[1])

        # Detect face in image
        faces_detected = face_cascade.detectMultiScale(img_np)

        for (x, y, w, h) in faces_detected:
            faces.append(img_np[y:y+h, x:x+w])
            ids.append(id)

    return faces, ids

print("🔍 Loading images…")
faces, ids = get_images_and_labels(dataset_path)

print("🧠 Training recognizer…")
recognizer.train(faces, np.array(ids))

# Save trained model
recognizer.save(f"{trainer_path}/trainer.yml")

print("✅ Training complete! Model saved as trainer/trainer.yml")
