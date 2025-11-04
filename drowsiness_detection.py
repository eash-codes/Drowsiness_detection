# okkk to first, I will import the OpenCV library because I need it to work with images and video from the webcam.
# I will load pre-trained models (Haar Cascades) to detect faces and eyes in the video frames.
# Then, I will capture video from my webcam so I can analyze live video.
# Then the main plan is to check if the eyes are open or closed in each video frame by analyzing the pixels.
# If the eyes stay closed for many frames in a row, I will assume the person is drowsy.
# When drowsiness is detected, I want to show a warning message on the screen (maybee play a audio later).
# I will also display rectangles around the face and eyes to help me see the detection working.
# The program will keep running and analyzing frames until I press 'q' to quit.
# This way, I can test how well the drowsiness detection works in real-time.
#okokokokokokokokokok

import cv2

# Loading the facee and eye detector 
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

camera = cv2.VideoCapture(0)  # Select the default webcam

closed_eye_count = 0
frames_to_alert = 15  # number of frames to trigger alert

def is_eye_open(eye_image):
    gray_eye = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)# conv to gs
    _, threshold_eye = cv2.threshold(gray_eye, 50, 255, cv2.THRESH_BINARY)# take the binary values
    white_pixel_count = cv2.countNonZero(threshold_eye)
    total_pixels = threshold_eye.size # simply len * width of the threshold eye
    white_ratio = white_pixel_count / total_pixels if total_pixels > 0 else 0
    return white_ratio > 0.15  # if more than 15% pixels white eye is open

while True:
    success, frame = camera.read() #success = 1 if frame read else false
    if not success:
        break

    frame = cv2.resize(frame, (640, 480))# resize the image for better and faster computation
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # conv the frame to gs 

    detected_faces = face_detector.detectMultiScale(gray_frame, 1.3, 5) #default values more like standard values fo rdetecting single face


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
