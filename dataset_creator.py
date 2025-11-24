import cv2
import os

# Create dataset folder
if not os.path.exists("dataset"):
    os.makedirs("dataset")

# Load face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam (DirectShow fixes many Windows camera issues)
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cam.isOpened():
    print("❌ Could not open webcam.")
    exit()

user_id = input("Enter User ID: ")
count = 0

while True:
    ret, frame = cam.read()

    if not ret:
        print("❌ Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face_img = gray[y:y+h, x:x+w]

        cv2.imwrite(f"dataset/User.{user_id}.{count}.jpg", face_img)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"Capturing {count}/100", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Dataset Creator", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if count >= 100:
        break

cam.release()
cv2.destroyAllWindows()

print("✅ Dataset captured successfully!")
