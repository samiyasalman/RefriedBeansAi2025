from ultralytics import YOLO
import cv2
import numpy as np
import tensorflow as tf
import time
from twilio.rest import Client

# Twilio credentials (sensitive)
account_sid = "asdfghjkl"
auth_token = "qwertyuiop"
twilio_number = "123456789" 
emergency_number = "987654321"

# Load YOLO model for hand gestures
hand_gesture_model = YOLO("/Users/xyz/Downloads/best3.pt")

# Load Keras model for facial expressions
expression_model = tf.keras.models.load_model("/Users/xyz/Downloads/expression_model.keras")

# Load OpenCV's pre-trained face detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Define class labels for 7 expressions (modify as needed)
class_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Function for predicting object detection
def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)
    return results

# Function for predicting AND drawing 
def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=3, text_thickness=2):
    results = predict(chosen_model, img, classes, conf=conf)
    for result in results:
        for box in result.boxes:
            # Bounding box
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (0, 0, 255), rectangle_thickness)
            # Label and confidence score
            cv2.putText(img, f"{result.names[int(box.cls[0])]}: {box.conf[0]:.2f}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), text_thickness)
    return img, results

def predict_expression(chosen_model, face_img):
    """Preprocess the face and predict the expression."""
    img_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img_resized = cv2.resize(img_gray, (chosen_model.input_shape[1], chosen_model.input_shape[2]))
    img_array = np.expand_dims(img_resized, axis=(0, -1)) / 255.0  # Normalize and add batch & channel dimension

    predictions = chosen_model.predict(img_array)
    class_id = np.argmax(predictions)  # Get the class with highest probability
    confidence = np.max(predictions)  # Get confidence score

    return class_labels[class_id], confidence

def call_emergency():
    """Call the emergency number using Twilio."""
    client = Client(account_sid, auth_token)
    call = client.calls.create(
        to=emergency_number,
        from_=twilio_number,
        twiml="<Response><Say>Emergency detected. Please respond.</Say></Response>"
    )

# Initialize video capture
cap = cv2.VideoCapture(0)

# State variables for hand gesture sequence
open_hand_detected = False
sequence_detected = False

# State variable for tracking fear detection
fear_detected_time = None
fear_timeout = 5  # Timeout in seconds

# State variable to track if emergency was contacted
emergency_contacted = False

while True:
    # Capturing each frame
    ret, frame = cap.read()
    if not ret:
        break

    if not sequence_detected:
        # Hand gesture detection
        result_img, results = predict_and_detect(hand_gesture_model, frame, classes=[], conf=0.5)

        # Check for "open_hand" and "closed_hand" sequence
        for result in results:
            for box in result.boxes:
                if result.names[int(box.cls[0])] == "open_hand":
                    open_hand_detected = True
                elif open_hand_detected and result.names[int(box.cls[0])] == "closed_hand":
                    sequence_detected = True

        # Display the PROCESSED frames
        cv2.imshow("Real-Time Object Detection", result_img)

    else:
        # Facial expression detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        fear_detected = False

        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, x:x + w]  # Crop the face

            # Predict expression
            expression, confidence = predict_expression(expression_model, face_img)

            # Draw bounding box around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display label and confidence
            cv2.putText(frame, f"{expression}: {confidence:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)

            # Check for "fear" with confidence > 0.5
            if expression == "Fear" and confidence > 0.3:
                fear_detected = True
                if fear_detected_time is None:
                    fear_detected_time = time.time()  # Record the time when fear is first detected
                    print("Fear detected! Calling emergency...")
                    call_emergency()
                    emergency_contacted = True  # Set emergency contacted flag

        # If no fear is detected, reset the fear detection time
        if not fear_detected:
            fear_detected_time = None

        # Check if 5 seconds have passed since the last fear detection
        if fear_detected_time and (time.time() - fear_detected_time) > fear_timeout:
            sequence_detected = False  # Revert to hand gesture detection
            open_hand_detected = False
            fear_detected_time = None
            emergency_contacted = False  # Reset emergency contacted flag
            print("No fear detected for 5 seconds. Reverting to hand gesture detection.")

        # Display "Emergency Contacted" on the top right of the screen if emergency was contacted
        if emergency_contacted:
            cv2.putText(frame, "Emergency Contacted", (frame.shape[1] - 300, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Display processed frame
        cv2.imshow("Real-Time Expression Classification", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()