import cv2
import os
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import simpledialog, messagebox

# Paths
dataset_path = "dataset"
trainer_path = "trainer"

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
if not os.path.exists(trainer_path):
    os.makedirs(trainer_path)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

# ---------------- Functions ---------------- #
def capture_dataset():
    user_id = simpledialog.askstring("Input", "Enter User ID:", parent=root)
    if not user_id:
        return

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    count = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            cv2.imwrite(f"{dataset_path}/User.{user_id}.{count}.jpg", face_img)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"Capturing {count}/100", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Capture Dataset", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 100:
            break

    cam.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Done", f"Dataset captured for User {user_id}")

def train_model():
    image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]
    faces = []
    ids = []

    for image_path in image_paths:
        img = Image.open(image_path).convert("L")
        img_np = np.array(img, 'uint8')
        id = int(os.path.split(image_path)[-1].split('.')[1])
        faces_detected = face_cascade.detectMultiScale(img_np)
        for (x, y, w, h) in faces_detected:
            faces.append(img_np[y:y+h, x:x+w])
            ids.append(id)

    recognizer.train(faces, np.array(ids))
    recognizer.save(f"{trainer_path}/trainer.yml")
    messagebox.showinfo("Done", "Training complete!")

def recognize_face():
    try:
        recognizer.read(f"{trainer_path}/trainer.yml")
    except:
        messagebox.showerror("Error", "Please train the model first!")
        return

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            id, conf = recognizer.predict(face_img)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"User {id} - {int(conf)}%", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

# ---------------- GUI ---------------- #
root = tk.Tk()
root.title("Face Recognition App")
root.geometry("300x200")

tk.Button(root, text="Capture Dataset", width=20, command=capture_dataset).pack(pady=10)
tk.Button(root, text="Train Model", width=20, command=train_model).pack(pady=10)
tk.Button(root, text="Recognize Face", width=20, command=recognize_face).pack(pady=10)

root.mainloop()
