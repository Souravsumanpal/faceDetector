import cv2
from random import randrange

# Load some pre-trained data on face frontals from opencv(haarcascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# to capture video from webcam
webcam = cv2.VideoCapture(0)

# Iterate forever over  frame
while True:

    ### Read the current frame
    successful_frame_read, frame = webcam.read()

    # Must convert to greyscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect face
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Draw a rectangle around the face
    for (x,y,w,h) in  face_coordinates:
        cv2.rectangle(frame, (x,y),(x+w,y+h), (0, 255, 0), 3)

    cv2.imshow('cleaver program face Detector', frame)
    key = cv2.waitKey(1) 


    ### Stop if Q is pressed
    if key == 81 or key == 113:
        break 


# Release the videocapture object
webcam.release      