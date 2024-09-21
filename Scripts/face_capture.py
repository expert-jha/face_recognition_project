import cv2
import os

# Initialize the face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def capture_faces(user_name, num_samples=1000):
    # Create a directory for the user if it doesn't exist
    dataset_dir = 'datasets'
    user_path = os.path.join(dataset_dir, user_name)
    
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        
    if not os.path.exists(user_path):
        os.makedirs(user_path)

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Set the frame width and height (optional: adjust according to your needs)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Capturing images for user:", user_name)
    
    count = 0
    while count < num_samples:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Draw rectangles around the faces and save the images
        for (x, y, w, h) in faces:
            # Extract face region
            face_img = frame[y:y+h, x:x+w]
            # Resize the face image to be rectangular (e.g., 100x100 pixels)
            face_img_resized = cv2.resize(face_img, (100, 100))
            
            # Save the image
            img_name = os.path.join(user_path, f'face_{count}.jpg')
            cv2.imwrite(img_name, face_img_resized)
            
            print(f"Image {count + 1} saved: {img_name}")
            count += 1
        
        # Display the resulting frame with face rectangles
        cv2.imshow('Face Capture', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    user_name = input("Enter user name: ")
    capture_faces(user_name)
