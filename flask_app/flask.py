from flask import Flask, render_template, Response
import cv2
import tensorflow as tf
import numpy as np

# Load the trained CNN model
model = tf.keras.models.load_model('../models/face_recognition_model.h5')

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the labels for classes (modify according to your model)
class_labels = {0: 'User1', 1: 'User2', 2: 'User3', 3: 'User4'}

# Initialize the Flask app
app = Flask(__name__)

# Function to generate video frames
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            face_img_resized = cv2.resize(face_img, (100, 100))
            face_img_array = np.expand_dims(face_img_resized, axis=0) / 255.0

            # Predict the class (user)
            predictions = model.predict(face_img_array)
            class_index = np.argmax(predictions[0])
            confidence = predictions[0][class_index]

            # Label the face
            if confidence > 0.5:
                label = class_labels.get(class_index, 'Unknown')
            else:
                label = 'Not Recognized'

            # Draw a rectangle around the face and add a label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Convert the frame to a format suitable for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route to serve the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to serve the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
