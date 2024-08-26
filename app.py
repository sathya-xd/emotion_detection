import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from collections import defaultdict
import matplotlib.pyplot as plt
import time

# Load the face classifier and model
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
emotion_counts = defaultdict(int)

# Global variable to control the interview state
interview_running = False

# Function to perform emotion detection
def detect_emotions(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            emotion_counts[label] += 1  # Increment the count for the detected emotion
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame

# Streamlit UI
st.title("Interview Performance Analyzer")
st.write("## Overview")
st.write("""
Welcome to the Interview Performance Analyzer. This application is designed to help you analyze and improve your performance in interviews by detecting and analyzing your emotions in real-time using your webcam.
""")

st.write("## Instructions")
st.write("""
1. Click the "Start Interview" button to begin the interview simulation.
2. The webcam will open, and the application will start detecting and analyzing your emotions.
3. Real-time feedback will be provided on the screen, showing the emotions detected during the interview.
4. After the interview, you can review the analysis of your emotional responses.
""")

# Placeholders for webcam and graph
frame_placeholder = st.empty()
graph_placeholder = st.empty()

# Function to start the interview (webcam feed and emotion detection)
def start_interview():
    global interview_running
    interview_running = True
    cap = cv2.VideoCapture(0)
    while cap.isOpened() and interview_running:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect_emotions(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Streamlit
        frame_placeholder.image(frame, use_column_width=True)
        
        # Update the graph
        fig, ax = plt.subplots()
        ax.bar(emotion_counts.keys(), emotion_counts.values())
        ax.set_xlabel('Emotions')
        ax.set_ylabel('Counts')
        ax.set_title('Counts of Emotions Detected During Interview')
        graph_placeholder.pyplot(fig)
        
        time.sleep(0.1)  # Add a small delay to limit the frame rate

    cap.release()
    cv2.destroyAllWindows()

# Function to stop the interview
def stop_interview():
    global interview_running
    interview_running = False

# Start the interview when the button is clicked
if st.button('Start Interview'):
    start_interview()

# Stop the interview when the button is clicked
if st.button('Stop Interview'):
    stop_interview()