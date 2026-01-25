import cv2
import pickle
import numpy as np
import os

# Setup
v = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
face_data = []
i = 0
max_faces = 100

# Get user input
name = input("Enter your name: ").strip()

print("📸 Capturing face data. Press 'q' to stop early.")

while True:
    ret, frame = v.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50))

        if len(face_data) < max_faces and i % 10 == 0:
            face_data.append(resized_img)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
        cv2.putText(frame, f"Samples: {len(face_data)}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        i += 1

    cv2.imshow("Live Feed", frame)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q') or len(face_data) >= max_faces:
        print("👋 Exiting face capture...")
        break

# Cleanup
v.release()
cv2.destroyAllWindows()

# Convert to numpy and reshape
face_data = np.asarray(face_data)
face_data = face_data.reshape(len(face_data), -1)

# Make sure data folder exists
os.makedirs("data", exist_ok=True)

# Load existing name and face data (if available)
name_file = "data/name.pkl"
face_file = "data/face_data.pkl"

# Load names
if os.path.exists(name_file):
    with open(name_file, 'rb') as f:
        names = pickle.load(f)
else:
    names = []

# Load faces
if os.path.exists(face_file):
    with open(face_file, 'rb') as f:
        faces = pickle.load(f)
else:
    faces = np.empty((0, 7500))  # 50 x 50 x 3 = 7500

# Append new data
names.extend([name] * len(face_data))
faces = np.append(faces, face_data, axis=0)

# Save updated data
with open(name_file, 'wb') as f:
    pickle.dump(names, f)

with open(face_file, 'wb') as f:
    pickle.dump(faces, f)

print(f"✅ Successfully saved {len(face_data)} samples for '{name}'")
