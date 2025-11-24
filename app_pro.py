import tkinter as tk
from tkinter import simpledialog, messagebox
import cv2
import os
import numpy as np
from PIL import Image

# ---------------- Paths ---------------- #
dataset_path = "dataset"
trainer_path = "trainer"

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
if not os.path.exists(trainer_path):
    os.makedirs(trainer_path)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

# ---------------- Global Camera Flag ---------------- #
camera_running = False

# ---------------- Functions ---------------- #
def flash_status(msg, times=4, interval=300):
    def _flash(count):
        if count > 0:
            status_label['fg'] = "red" if status_label['fg'] == "yellow" else "yellow"
            status_label.after(interval, _flash, count-1)
        else:
            status_label['fg'] = "yellow"
            status_label['text'] = msg
    _flash(times)

def update_status(msg):
    status_label.config(text=msg)
    root.update_idletasks()

def capture_dataset():
    global camera_running
    user_id = simpledialog.askstring("Input", "Enter User ID:", parent=root)
    if not user_id:
        return

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    count = 0
    camera_running = True
    flash_status(f"Capturing dataset for User {user_id}...")

    while camera_running:
        ret, frame = cam.read()
        if not ret:
            update_status("❌ Failed to open camera!")
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

        cv2.imshow("Dataset Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 100:
            break

    camera_running = False
    cam.release()
    cv2.destroyAllWindows()
    update_status(f"✅ Dataset captured for User {user_id}")
    messagebox.showinfo("Done", f"Dataset captured for User {user_id}")

def train_model():
    flash_status("🔍 Loading images...")
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

    update_status("🧠 Training model...")
    recognizer.train(faces, np.array(ids))
    recognizer.save(f"{trainer_path}/trainer.yml")
    update_status("✅ Training complete!")
    messagebox.showinfo("Done", "Training complete!")

def recognize_face():
    global camera_running
    try:
        recognizer.read(f"{trainer_path}/trainer.yml")
    except:
        messagebox.showerror("Error", "Please train the model first!")
        return

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    camera_running = True
    flash_status("🔍 Recognizing face...")

    while camera_running:
        ret, frame = cam.read()
        if not ret:
            update_status("❌ Failed to open camera!")
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

    camera_running = False
    cam.release()
    cv2.destroyAllWindows()
    update_status("✅ Recognition stopped.")

def clear_dataset():
    if messagebox.askyesno("Confirm", "Do you want to delete all dataset images?"):
        for file in os.listdir(dataset_path):
            os.remove(os.path.join(dataset_path, file))
        update_status("✅ Dataset cleared!")

def stop_camera():
    global camera_running
    if camera_running:
        camera_running = False
        update_status("🛑 Camera stopped by user")

# ---------------- GUI ---------------- #
root = tk.Tk()
root.title("Face Recognition Professional App")
root.geometry("400x550")
root.configure(bg="#2C3E50")

title = tk.Label(root, text="Face Recognition App",
                 font=("Helvetica", 20, "bold"),
                 bg="#2C3E50", fg="white")
title.pack(pady=10)

btn_frame = tk.Frame(root, bg="#2C3E50")
btn_frame.pack(pady=20)

# -------------- BIG BUTTON STYLING -------------- #
btn_style = {
    "width": 25,
    "height": 3,
    "bg": "#3498DB",
    "fg": "white",
    "font": ("Helvetica", 14, "bold"),
    "relief": "flat",
    "activebackground": "#2980B9"
}

clear_btn_style = btn_style.copy()
clear_btn_style["bg"] = "#E74C3C"
clear_btn_style["activebackground"] = "#C0392B"

# ----------- Hover Animations (Updated Size) ----------- #
def animate_enter(e):
    e.widget['bg'] = "#1ABC9C"
    e.widget.config(width=28, height=4)

def animate_leave(e, color):
    e.widget['bg'] = color
    e.widget.config(width=25, height=3)

buttons = []

# Create all buttons
btn1 = tk.Button(btn_frame, text="Capture Dataset", command=capture_dataset, **btn_style)
btn1.pack(pady=7)
buttons.append((btn1, "#3498DB"))

btn2 = tk.Button(btn_frame, text="Train Model", command=train_model, **btn_style)
btn2.pack(pady=7)
buttons.append((btn2, "#3498DB"))

btn3 = tk.Button(btn_frame, text="Recognize Face", command=recognize_face, **btn_style)
btn3.pack(pady=7)
buttons.append((btn3, "#3498DB"))

btn4 = tk.Button(btn_frame, text="Clear Dataset", command=clear_dataset, **clear_btn_style)
btn4.pack(pady=7)
buttons.append((btn4, "#E74C3C"))

btn5 = tk.Button(btn_frame, text="Stop Camera", command=stop_camera, **btn_style)
btn5.pack(pady=7)
buttons.append((btn5, "#3498DB"))

btn_exit = tk.Button(btn_frame, text="Exit", command=root.destroy, **btn_style)
btn_exit.pack(pady=10)
buttons.append((btn_exit, "#3498DB"))

# Apply animations
for btn, color in buttons:
    btn.bind("<Enter>", animate_enter)
    btn.bind("<Leave>", lambda e, col=color: animate_leave(e, col))

# ---------------- Status Label ---------------- #
status_label = tk.Label(root, text="Ready", font=("Helvetica", 14),
                        bg="#2C3E50", fg="yellow")
status_label.pack(pady=10)

root.mainloop()
