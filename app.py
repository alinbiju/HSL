import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# Load the pre-trained model
model = load_model('action.h5')

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic

# Streamlit configuration
st.set_page_config(page_title="Sign Language Prediction", page_icon="✋")

# Streamlit layout
st.title("Sign Language Prediction")
st.markdown("Use the camera to perform sign language, and the predicted sentence will be displayed below.")

# Create placeholders
stframe = st.empty()
sentence_placeholder = st.empty()

# Function to perform sign language prediction
def predict_sign_language():
    cap = cv2.VideoCapture(0)
    sequence = []
    sentence = []
    threshold = 0.7

    res = None  # Initialize res here

    # Set MediaPipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            image, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)
            sequence.insert(0, keypoints)
            sequence = sequence[:30]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]

            if res is not None and np.argmax(res) < len(actions) and res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5:
                sentence = sentence[-5:]

            image = prob_viz(res, actions, image, colors)
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 225, 225), 2, cv2.LINE_AA)
            stframe.image(image, channels="BGR")

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# Function to run MediaPipe
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Function to display probabilities
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 255, 255), 2, cv2.LINE_AA)
    return output_frame

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])


st.button("Start Predicting", on_click=predict_sign_language)

# Streamlit app
# if __name__ == "__main__":
#     st.run()
