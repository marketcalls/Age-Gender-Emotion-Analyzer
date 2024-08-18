import cv2
import numpy as np
from keras.models import load_model
from keras.utils import img_to_array
import os


def load_models():
    """Load the pre-trained models for age, gender and emotion prediction."""
    age_model_path = './model/age_model.h5'
    gender_model_path = './model/gender_model.h5'
    emotion_model_path = './model/emotion_model.h5'

    age_model = load_model(age_model_path)
    gender_model = load_model(gender_model_path)
    emotion_model = load_model(emotion_model_path)

    return age_model, gender_model, emotion_model

def predict(frame, age_model, gender_model, emotion_model):
    """Predict the age, gender, and emotion for the given frame."""
    age_ranges = ['1-2', '3-9', '10-20', '21-27', '28-45', '46-65', '66-116']
    gender_ranges = ['male', 'female']
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

        face_gray = gray[y:y+h, x:x+w]

        # Emotion prediction
        roi_gray = cv2.resize(face_gray, (48, 48), interpolation=cv2.INTER_AREA)
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            emotion_prediction = emotion_model.predict(roi)[0]
            emotion_label = emotion_labels[emotion_prediction.argmax()]
        else:
            emotion_label = 'No Face'

        # Gender prediction
        gender_img = cv2.resize(face_gray, (100, 100), interpolation=cv2.INTER_AREA)
        gender_input = np.expand_dims(gender_img, axis=[0, -1])
        gender_label = gender_ranges[np.argmax(gender_model.predict(gender_input))]

        # Age prediction
        age_img = cv2.resize(face_gray, (200, 200), interpolation=cv2.INTER_AREA)
        age_input = np.expand_dims(age_img, axis=[0, -1])
        age_label = age_ranges[np.argmax(age_model.predict(age_input))]

        # Combine predictions
        output_str = f"{gender_label}, {age_label}, {emotion_label}"
        cv2.putText(frame, output_str, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    return frame

def main():
    """Main function to execute continuous prediction from webcam."""
    age_model, gender_model, emotion_model = load_models()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = predict(frame, age_model, gender_model, emotion_model)

        cv2.imshow('Age, Gender, and Emotion Prediction', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()