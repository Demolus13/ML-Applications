import os
import cv2
import torch
import streamlit as st
from streamlit_option_menu import option_menu

from Models import HGRModel, landmarks_to_list, normalize_landmarks

import mediapipe as mp
mp_drawings = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Set page configuration
st.set_page_config(page_title="ML Application Models",
                   layout="centered")

# loading the saved models
Models = {
    'HGR': {},
}

# Initialize MediaPipe Hands
HandLandmarker = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5
)

Models_dir = "Models"
for model_name in os.listdir(Models_dir):
    with open(os.path.join(Models_dir, model_name), 'rb') as file:
        if model_name.startswith('HGR'):
            model = HGRModel(42, 4)
            model.load_state_dict(torch.load(file, map_location=torch.device('cpu')))
            Models['HGR'][model_name] = model

# sidebar for navigation
with st.sidebar:
    selected = option_menu('ML Models',
                           [
                               'Hand Gesture Recognition',
                            ],
                           default_index=0)


# Decison Tree Page
if selected == 'Hand Gesture Recognition':

    # page title
    st.title('Hand Gesture Recognition Model')

    # dataset link
    # st.markdown(
    #     """
    #     <a href="https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic" target="_blank">
    #         <button style='background-color: #262730;
    #         border: 0px;
    #         border-radius: 10px;
    #         color: white;
    #         padding: 10px 15px;
    #         text-align: center;
    #         text-decoration: none;
    #         font-size: 16px;
    #         margin-bottom: 1rem;
    #         cursor: pointer;'>Breast Cancer Dataset</button>
    #     </a>
    #     """, 
    #     unsafe_allow_html=True,
    # )

    classes = {
        0: "Right hand open",
        1: "Left hand open",
        2: "Right hand close",
        3: "Left hand close"
    }

    classes_cols = st.columns(4)
    for i, (key, value) in enumerate(classes.items()):
        with classes_cols[i]:
            st.write(f"{key}: {value}")

    if 'run' not in st.session_state:
        st.session_state.run = False

    cols = st.columns(2)
    with cols[0]:
        st.session_state.run = st.button('Run')
    with cols[1]:
        st.session_state.run = not st.button('Stop')

    FRAME_WINDOW = st.empty()
    cap = cv2.VideoCapture(0)

    if st.session_state.run:
        with HandLandmarker as landmarker:
            while st.session_state.run:
                success, frame = cap.read()
                if not success:
                    continue

                # Convert the BGR image to RGB.
                frame = cv2.flip(frame, 1)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect the hand landmarks
                frame.flags.writeable = False
                results = landmarker.process(frame)

                # Draw the hand annotations on the image.
                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawings.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawings.DrawingSpec(color=(97, 137, 48), thickness=2, circle_radius=4),
                            mp_drawings.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                        )

                        # Convert the landmarks to a list wrt the image
                        landmarks = landmarks_to_list(frame, results.multi_hand_landmarks)

                        # Normalize the landmarks
                        landmarks = normalize_landmarks(landmarks).reshape(1, -1)
                        Y_pred = model(landmarks)
                        pred_class = torch.argmax(Y_pred, axis=1).item()

                        # Annotate the predicted class on the screen
                        cv2.putText(frame, classes[pred_class], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                FRAME_WINDOW.image(frame)

        cap.release()

    # code block
    st.title('Notebook')
    code = '''
    # Initialize MediaPipe Hands
    HandLandmarker = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.5
    )

    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    label = 0
    count = 0

    with HandLandmarker as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # the BGR image to RGB.
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Dectect the hand landmarks
            frame.flags.writeable = False
            results = landmarker.process(frame)

            # Draw the hand annotations on the image.
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            key = cv2.waitKey(5) & 0xFF
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawings.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawings.DrawingSpec(color=(97, 137, 48), thickness=2, circle_radius=4),
                        mp_drawings.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                    )
                
                if key == 13:
                    # Convert the landmarks to a list wrt the image
                    landmarks = landmarks_to_list(frame, results.multi_hand_landmarks)

                    # Normalize the landmarks
                    landmarks = normalize_landmarks(landmarks).flatten()

                    # Append the label to the landmarks and store in the dataset
                    dataset.append(torch.cat((landmarks, torch.tensor([label]))))
                    print(f"{count} - Gesture Labelled: {label}")

            cv2.imshow('MediaPipe Hands', frame)
            if key == 27: # ESC
                break

    cap.release()
    # cv2.destroyAllWindows()
    '''
    st.code(code, language='python')
    code = '''
    class HGRModel(nn.Module):
        """
        A MLP Model for Hand Gesture Recognition
        """
        def __init__(self, in_features, out_features):
            super(HGRModel, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(in_features, 16),
                nn.ReLU(),
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, out_features),
                nn.Softmax(dim=1)  # Add softmax layer
            )
            
        def forward(self, x):
            return self.model(x)
        
        def fit(self, X, Y, epochs=1000, lr=0.01):
            """
            X: torch.Tensor of shape (n_samples, n_features)
            Y: torch.Tensor of shape (n_samples, n_channels)
            epochs: int, the number of epochs
            """
            criteria = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            
            for epoch in range(epochs):
                # Forward pass
                preds = self.forward(X)
                loss = criteria(preds, Y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Print the loss
                if (epoch+1) % 10 == 0:
                    print(f'Epoch {epoch+1} Loss: {loss.item()}')
                    print("\\n----------------------------------------------------\\n")

        def save_model(self, file_path):
            """
            Save the model to a file.
            """
            torch.save(self.state_dict(), file_path)

        def load_model(self, file_path):
            """
            Load the model from a file.
            """
            self.load_state_dict(torch.load(file_path))
    '''
    st.code(code, language='python')
