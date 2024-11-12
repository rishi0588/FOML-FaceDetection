import streamlit as st
import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
 
# Load the pre-trained face mask detection model
model = load_model("C:\\Users\\RISHI\\Desktop\\sem 5\\FOML\\streamlit\\model.keras")
 
# Function to detect mask from an image
def detect_face_mask(img):
    y_pred = model.predict(img.reshape(1, 224, 224, 3))
    return y_pred[0][0]
 
# Function to draw label on the image
def draw_label(img, text, pos, bg_color):
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, cv2.FILLED)
    end_x = pos[0] + text_size[0][0] + 2
    end_y = pos[1] + text_size[0][1] - 2
    cv2.rectangle(img, pos, (end_x, end_y), bg_color, cv2.FILLED)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
 
# Function to detect faces in an image using Haar Cascade
def detect_face(img):
    haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    coods = haar_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return coods
 
# Start the Streamlit app
st.title("Real-Time Face Mask Detection")
st.write("This app detects whether a person is wearing a mask in real-time using your webcam.")
 
# Use Streamlit's webcam capture
stframe = st.empty()
 
cap = cv2.VideoCapture(0)
 
while True:
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to grab frame")
        break
 
    # Resize the frame to 224x224 as required by the model
    img = cv2.resize(frame, (224, 224))
 
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    # Detect faces
    faces = detect_face(gray)
 
    # Draw rectangles around detected faces and predict mask
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw rectangle around face
        face = frame[y:y + h, x:x + w]
        face = cv2.resize(face, (224, 224))
 
        # Detect mask (0: No Mask, 1: Mask)
        y_pred = detect_face_mask(face)
        if y_pred == 1.:
            draw_label(frame, "No Mask", (x, y), (0, 0, 255))  # Green for mask
        else:
            draw_label(frame, "Mask", (x, y), (0, 255, 0))  # Red for no mask
 
    # Display the result in the Streamlit app
    stframe.image(frame, channels="BGR", use_column_width=True)
 
    # Break the loop if the user presses 'x'
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
 
cap.release()
cv2.destroyAllWindows()