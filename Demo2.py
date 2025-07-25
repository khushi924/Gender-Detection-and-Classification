import cv2
import tkinter as tk
from tkinter import filedialog,messagebox, PhotoImage
from threading import Thread

import time
import numpy as np

# Load models
face_pbtxt = "models/opencv_face_detector.pbtxt"
face_pb = "models/opencv_face_detector_uint8.pb"
gender_prototxt = "models/gender_deploy.prototxt"
gender_model = "models/gender_net.caffemodel"
MODEL_MEAN_VALUES = [104, 117, 123]
gender_classes = ['Male', 'Female']

# Load DNNs
face_net = cv2.dnn.readNet(face_pb, face_pbtxt)
gender_net = cv2.dnn.readNet(gender_model, gender_prototxt)

# Gender Detection Function
def detect_gender(frame):
    img_cp = frame.copy()
    h, w = img_cp.shape[:2]
    blob = cv2.dnn.blobFromImage(img_cp, 1.0, (300, 300), MODEL_MEAN_VALUES, swapRB=False, crop=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)

            face_region = img_cp[max(0, y1-15):min(y2+15, h-1), max(0, x1-15):min(x2+15, w-1)]

            if face_region.shape[0] == 0 or face_region.shape[1] == 0:
                continue

            try:
                face_blob = cv2.dnn.blobFromImage(face_region, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=True)
                gender_net.setInput(face_blob)
                gender_pred = gender_net.forward()
                gender = gender_classes[gender_pred[0].argmax()]
                # Draw results
                cv2.rectangle(img_cp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_cp, gender, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            except Exception as e:
                print("Face detection error:", e)

    return img_cp

# --- Detection Modes ---
def detect_image():
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if not path:
        return
    img = cv2.imread(path)
    result = detect_gender(img)
    cv2.imshow("Image Gender Detection", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_video():
    path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    if not path:
        return
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result = detect_gender(frame)
        cv2.imshow("Video Gender Detection", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.02)  # ~50 FPS
    cap.release()
    cv2.destroyAllWindows()

def detect_webcam():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result = detect_gender(frame)
        cv2.imshow("Webcam Gender Detection", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.03)  # ~30 FPS
    cap.release()
    cv2.destroyAllWindows()
    import sys



# --- GUI App ---
def start_gui():
    root = tk.Tk()
    root.title("Gender Classifier App")
    root.config(bg="Black")
    root.geometry("1080x1080")
    root.resizable(False, False)
    root.attributes("-fullscreen",True)
    try:
        logo = PhotoImage(file="logo.png")
        tk.Label(root, image=logo, bg="#e9eff5").pack(pady=15)
    
    except Exception:
        tk.Label(root, text="Gender-Genie", font=("Helvetica", 28, "bold"), bg="#e9eff5").pack(pady=20)
   
    tk.Label(root, text="Gender-Genie", font=("Algerian", 16,"bold"), bg="lightblue").pack(pady=0)
    tk.Label(root, text="Gender Detection and Classification", font=("Harlow Solid Italic", 12,), bg="lightblue").pack(pady=15)

    tk.Button(root, text="Detect from Image", width=25,
              command=lambda: Thread(target=detect_image).start()).pack(pady=10)
    tk.Button(root, text="Detect from Video", width=25,
              command=lambda: Thread(target=detect_video).start()).pack(pady=10)
    tk.Button(root, text="Detect from Webcam", width=25,
              command=lambda: Thread(target=detect_webcam).start()).pack(pady=10)
    

    tk.Button(root, text="Exit", width=25, command=root.quit).pack(pady=20)
    root.mainloop()

    

if __name__ == "__main__":
    start_gui()
