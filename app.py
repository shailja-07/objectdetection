import streamlit as st
import cv2
from ultralytics import YOLO
import pyttsx3


tts = pyttsx3.init()
tts.setProperty('rate', 150)


model = YOLO("yolov8n.pt")

st.title("Object Detection")
FRAME_WINDOW = st.image([])

start_button = st.button("Start Camera")
stop_button = st.button("Stop Camera")
spoken_labels = set()

run = start_button and not stop_button

if run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam.")
        st.stop()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to grab frame.")
            break

        height, width, _ = frame.shape
        results = model(frame)
        annotated_frame = results[0].plot()
        FRAME_WINDOW.image(annotated_frame, channels="BGR")

        detections = results[0].boxes
        if detections is not None:
            for box in detections:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, xyxy)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                
                horiz = "left" if cx < width / 3 else "right" if cx > 2 * width / 3 else "center"
                vert = "top" if cy < height / 3 else "bottom" if cy > 2 * height / 3 else "middle"
                position = f"{vert} {horiz}"

                if conf >= 0.70 and (label, position) not in spoken_labels:
                    msg = f"Detected {label} at {position}."
                    st.write(msg)
                    tts.say(msg)
                    tts.runAndWait()
                    spoken_labels.add((label, position))

        if stop_button:
            break

    cap.release()
    st.success("Camera stopped.")
