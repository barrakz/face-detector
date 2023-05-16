import cv2

trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# Capture video from webcam.
webcam = cv2.VideoCapture(0)

# Iterate forever over frames.
while True:

    # Read current frame.
    successful_frame_read, frame = webcam.read()

    # Convert to greyscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces.
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Draw rectagles around the faces.
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Face Detector App', frame)
    key = cv2.waitKey(1)

    # Stop if Q key is pressed.
    if key == 81 or key == 113:
        break

webcam.release()


#


print("Code completed")
