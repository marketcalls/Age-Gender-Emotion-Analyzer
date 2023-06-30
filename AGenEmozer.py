import cv2
import numpy as np
from keras.models import load_model
import argparse

# Define the model paths
AGE_MODEL_PATH = 'age_model.h5'
GENDER_MODEL_PATH = 'gender_model.h5'
EMOTION_MODEL_PATH = 'emotion_model.h5'

# Define the input shape for the models
INPUT_SHAPE = (200, 200, 1)

# Define the labels for gender and emotion
GENDER_LABELS = ['Male', 'Female']
EMOTION_LABELS = ['Positive', 'Negative', 'Neutral']

# Define the age mapping
def map_age(age):
    if 1 <= age <= 2:
        return 0
    elif 3 <= age <= 9:
        return 1
    elif 10 <= age <= 20:
        return 2
    elif 21 <= age <= 27:
        return 3
    elif 28 <= age <= 45:
        return 4
    elif 46 <= age <= 65:
        return 5
    else:
        return 6

# Define the emotion mapping
def map_emotion(label):
    if label in [4, 6]:
        return 0  # Positive emotion
    elif label in [0, 5]:
        return 1  # Negative emotion
    else:
        return 2  # Neutral emotion

class FaceAnalyzer:
    """Class for predicting age, gender, and emotion from an image or webcam feed."""

    def __init__(self, age_model_path, gender_model_path, emotion_model_path):
        self.age_model = load_model(age_model_path)
        self.gender_model = load_model(gender_model_path)
        self.emotion_model = load_model(emotion_model_path)
        self.input_shape = INPUT_SHAPE
        self.gender_labels = GENDER_LABELS
        self.emotion_labels = EMOTION_LABELS

        # Load Haar cascade classifier for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def preprocess_image(self, image):
        """Preprocess the image for model prediction."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return None

        # Extract the first detected face
        (x, y, w, h) = faces[0]
        face_roi = gray[y:y + h, x:x + w]

        resized = cv2.resize(face_roi, self.input_shape[:2])
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, *self.input_shape))
        return reshaped

    def predict_age_gender_emotion(self, image):
        """Predict age, gender, and emotion from the given image."""
        preprocessed_image = self.preprocess_image(image)

        if preprocessed_image is None:
            return None

        age_prediction = self.age_model.predict(preprocessed_image)
        predicted_age = map_age(np.argmax(age_prediction))

        gender_prediction = self.gender_model.predict(preprocessed_image)
        predicted_gender = self.gender_labels[np.argmax(gender_prediction)]

        emotion_prediction = self.emotion_model.predict(preprocessed_image)
        predicted_emotion = self.emotion_labels[map_emotion(np.argmax(emotion_prediction))]

        return predicted_age, predicted_gender, predicted_emotion

    def predict_image(self, image_path):
        """Predict age, gender, and emotion from an image."""
        image = cv2.imread(image_path)
        predicted_age, predicted_gender, predicted_emotion = self.predict_age_gender_emotion(image)

        if predicted_age is None:
            print("No face detected in the image.")
            return

        prediction_text = f'Age: {predicted_age}  Gender: {predicted_gender}  Emotion: {predicted_emotion}'
        cv2.putText(image, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def predict_webcam(self):
        """Predict age, gender, and emotion from the webcam feed."""
        capture = cv2.VideoCapture(0)
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            cv2.imshow('Webcam', frame)

            predicted_age, predicted_gender, predicted_emotion = self.predict_age_gender_emotion(frame)

            if predicted_age is not None:
                prediction_text = f'Age: {predicted_age}  Gender: {predicted_gender}  Emotion: {predicted_emotion}'
                cv2.putText(frame, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Webcam', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Predict age, gender, and emotion from an image or webcam.')
    parser.add_argument('--image', type=str, help='Path to the image file')
    args = parser.parse_args()

    # Create the FaceAnalyzer instance
    analyzer = FaceAnalyzer(AGE_MODEL_PATH, GENDER_MODEL_PATH, EMOTION_MODEL_PATH)

    # Check if image path is provided
    if args.image:
        analyzer.predict_image(args.image)
    else:
        analyzer.predict_webcam()
