import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from collections import defaultdict
import matplotlib
matplotlib.use('TkAgg')

# Define file paths
cascade_path = 'D:/projects/KIRAN_YARASHI_4NN21CS061_SATHYA_PRAMOD_4NN21CS047(CG PROJECT)/project_file/haarcascade_frontalface_default.xml'
model_path = 'D:/projects/KIRAN_YARASHI_4NN21CS061_SATHYA_PRAMOD_4NN21CS047(CG PROJECT)/project_file/model.h5'

# Check if the files exist
if not os.path.exists(cascade_path):
    raise FileNotFoundError(f"Cannot find file: {cascade_path}")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Cannot find file: {model_path}")

# Load the face classifier and model
face_classifier = cv2.CascadeClassifier(cascade_path)
classifier = load_model(model_path)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize a dictionary to count emotions
emotion_counts = defaultdict(int)

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            emotion_counts[label] += 1  # Increment the count for the detected emotion
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Plot the counts of each emotion
plt.bar(emotion_counts.keys(), emotion_counts.values())
plt.xlabel('Emotions')
plt.ylabel('Counts')
plt.title('Counts of Emotions ')
plt.show()
