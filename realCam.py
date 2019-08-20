import face_recognition
import cv2
import numpy as np
import math
from sklearn import neighbors
import os
import os.path
import pickle
from utils.annModel import Model

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
count = 0 

model = Model()
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        
        # Load image file and find face locations
        X_face_locations = face_recognition.face_locations(rgb_small_frame)
        # print(X_face_locations)

        # If no faces are found in the image, return an empty result.
        if len(X_face_locations) == 0:
            continue

        # Find encodings for faces in the test iamge
        faces_encodings = face_recognition.face_encodings(rgb_small_frame, X_face_locations)

        # Use the KNN model to find the best matches for the test face
        names = model.predict(faces_encodings)
        # Predict classes and remove classifications that aren't within the threshold
        infor = [(loc, pred) for pred, loc in zip(names, X_face_locations)]

        # Display the results
        for (top, right, bottom, left), name in infor:
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    if count % 5 == 0:
        process_this_frame = not process_this_frame

    count += 1

    # Display the resulting image
    cv2.imshow('Video', frame)

    
    
    
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()