import face_recognition
import cv2
import numpy as np
import os
from PIL import Image, ImageDraw
from utils.annModel import Model

DIR_TEST = 'TEST_IMG'

# Initialize some variables

model = Model()

for image_file in os.listdir(DIR_TEST):

    full_file_path = os.path.join(DIR_TEST, image_file)
    unknown_image = face_recognition.load_image_file(full_file_path)

   
    # Load image file and find face locations
    X_face_locations = face_recognition.face_locations(unknown_image, model = 'cnn')
    # print(X_face_locations)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        continue

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(unknown_image, X_face_locations)

    pil_image = Image.fromarray(unknown_image).convert("RGB")
    # Create a Pillow ImageDraw Draw instance to draw with
    draw = ImageDraw.Draw(pil_image)
    
    # Use the KNN model to find the best matches for the test face
    names = model.predict(faces_encodings)
    # Predict classes and remove classifications that aren't within the threshold
    infor = [(loc, pred) for pred, loc in zip(names, X_face_locations)]

    # Display the results
    for (top, right, bottom, left), name in infor:

        draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0), width = 3)

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)

        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
    
    del draw

    # Display the resulting image
    pil_image.show()
