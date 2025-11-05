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
import winsound
# Loading the facee and eye detector 
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

camera = cv2.VideoCapture(0)  # Select the default webcam

closed_eye_count = 0
frames_to_alert = 10  # number of frames to trigger alert

def is_eye_open(eye_image):
    gray_eye = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)# conv to gs
    _, threshold_eye = cv2.threshold(gray_eye, 50, 255, cv2.THRESH_BINARY)# take the binary values
    white_pixel_count = cv2.countNonZero(threshold_eye)
    total_pixels = threshold_eye.size # simply len * width of the threshold eye
    white_ratio = white_pixel_count / total_pixels if total_pixels > 0 else 0
   #print(f"White ratio in eye: {white_ratio:.3f}")  # New line to debug
    return white_ratio > 0.15  # if more than 15% pixels white eye is open

while True:
    success, frame = camera.read() #success = 1 if frame read else false
    if not success:
        break

    frame = cv2.resize(frame, (640, 480))# resize the image for better and faster computation
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # conv the frame to gs 

    detected_faces = face_detector.detectMultiScale(gray_frame, 1.3, 5) #default values more like standard values fo rdetecting single face

    if len(detected_faces) > 0:#atleast one face detected
        x, y, width, height = detected_faces[0]#use coords
        face_gray = gray_frame[y:y+height, x:x+width]# get grayscale face area
        face_color = frame[y:y+height, x:x+width]# get colored one too

        detected_eyes = eye_detector.detectMultiScale(face_gray)# detectt eyess in face region
        # print number of eyes for debugging (can delete later)
        #print(f"Eyes detected: {len(detected_eyes)}")

        if len(detected_eyes) >= 2: # if two or more eyes detected
            eyes_open = True # assumed both to be open
            for (eye_x, eye_y, eye_w, eye_h) in detected_eyes[:2]: #process the two eyes
                eye_img = face_color[eye_y:eye_y+eye_h, eye_x:eye_x+eye_w]
                eye_is_open = is_eye_open(eye_img) # check if eyes is open based on whitee pixwels
                color_box = (0, 255, 0) if eye_is_open else (0, 0, 255) #rect around eyes
                cv2.rectangle(face_color, (eye_x, eye_y), (eye_x+eye_w, eye_y+eye_h), color_box, 2)

                if not eye_is_open:# if closed set eyes open to false
                    eyes_open = False

            if eyes_open:
                closed_eye_count = 0
            else:
                closed_eye_count += 1
            
            #print(f"Closed eye frames: {closed_eye_count}")

            if closed_eye_count > frames_to_alert:# show alert
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2)
        else:
            closed_eye_count = 0

        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2) # blue rect arund face
    else:
        closed_eye_count = 0 # reset counter

    cv2.imshow('Drowsiness Detector', frame) # show processed frame

    if cv2.waitKey(1) & 0xFF == ord('q'):# break loop on 1
        break

camera.release()
cv2.destroyAllWindows()
