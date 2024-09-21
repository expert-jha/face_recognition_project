import cv2
import tensorflow as tf
import numpy as np

# Load the trained CNN model
model = tf.keras.models.load_model('models/face_recognition_model.h5')

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the labels for classes
class_labels = {0: 'Ashish', 1: 'Govind'}  # Update with your actual class indices

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# Create a named window for full screen mode
cv2.namedWindow('Face Recognition', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Face Recognition', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_img = frame[y:y+h, x:x+w]
            # Resize and preprocess the face image
            face_img_resized = cv2.resize(face_img, (100, 100))  # Resize to match the model's input shape
            face_img_array = np.expand_dims(face_img_resized, axis=0) / 255.0  # Normalize the image

            # Predict using the trained model
            predictions = model.predict(face_img_array)
            class_index = np.argmax(predictions[0])  # Get the index of the highest probability
            
            # Get the label for the class index
            label = class_labels.get(class_index, 'Unknown')
            
            # Draw rectangle and label on the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        # Display video feed in full screen
        cv2.imshow('Face Recognition', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
