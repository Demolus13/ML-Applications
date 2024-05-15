"""
Importing necessary libraries
"""

import torch
import torch.nn as nn

"""
Defining Utiliy Functions
"""

def landmarks_to_list(image, multi_hand_landmarks) -> torch.Tensor:
    """
    image: The image on which the landmarks are detected
    multi_hand_landmarks: The landmarks of the hand

    Returns: torch.Tensor: The tensor of landmarks [x, y]
    """

    image_height, image_width, _ = image.shape

    landmarks_list = []
    if multi_hand_landmarks:
        for hand_landmarks in multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                # orgin is top left corner (0, 0)
                landmarks_list.append([min(int(landmark.x * image_width), image_width - 1), -min(int(landmark.y * image_height), image_height - 1)])

    return torch.tensor(landmarks_list)

def normalize_landmarks(landmarks: torch.Tensor) -> torch.Tensor:
    """
    landmarks: torch.Tensor: The landmarks of the hand

    Returns: torch.Tensor: The normalized landmarks
    """

    landmarks = landmarks - landmarks[0]
    landmarks = landmarks / torch.max(landmarks)

    return landmarks

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
                print("\n----------------------------------------------------\n")

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