# ---- visual feature extraction using VGG19 ----
# v1: 4/24/2026
# Szymon Wabno

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2

class QRFeatureExtractor(nn.Module):
    """
    builds custom VGG19 architecture as described in the 
    research paper
    """
    def __init__(self):
        super(QRFeatureExtractor, self).__init__()
        
        # Adapter Layer (going from 1 to 3 layers)
        # Parameters: 1 input channel (grayscale), 3 output channels, 3x3 kernel, stride 1, padding 0
        self.conv2d_adapter = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=0)
        self.relu = nn.ReLU()
        
        # VGG19 Backbone
        # load standard VGG19,only keep features portion, starts with blank weights
        vgg19 = models.vgg19(weights=None) 
        self.vgg19_features = vgg19.features
        self.adaptive_pool = vgg19.avgpool # outputs to a standard 7x7 grid
        
        # Output Layer
        # with 7x7 grid w/512 layers --> get 25088 neurons
        # compress to final 1024 unique features
        self.fc = nn.Linear(in_features=25088, out_features=1024, bias=True)

    def forward(self, x):
        """
        defines exact path image takes through network
        """
        # Pass through the adapter
        x = self.conv2d_adapter(x)
        x = self.relu(x)
        
        # Pass through VGG19
        x = self.vgg19_features(x)
        x = self.adaptive_pool(x)
        
        # Flatten the data from 3D grid to 1D list of nums
        x = torch.flatten(x, 1)
        
        # Pass through final layer
        x = self.fc(x)
        return x

# ----------------------------
# Helper function for app.py
# -----------------------------

# Initialize the model once
model = QRFeatureExtractor()
model.eval() # Set to eval mode (turns off training features)

def extract_features(cv2_image):
    """
    Can call this from app.py, takes a cropped webcam image, 
    preps it, and returns 1024 feature vector
    """
    # convert the OpenCV image (NumPy array) to a PIL Image in Grayscale ('L')
    pil_image = Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY))
    
    # define pre-processing steps required by the paper
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)), # Force it to 224x224 pixels
        transforms.ToTensor(),         # Convert to a PyTorch Tensor
    ])
    
    # apply preprocessing and add a "batch" dimension (PyTorch expects [batch, channel, height, width])
    input_tensor = preprocess(pil_image).unsqueeze(0) 
    
    # run image through model w/o calculating gradients to save mem
    with torch.no_grad():
        feature_vector = model(input_tensor)
        
    # return 1D list of 1024 numbers
    return feature_vector.numpy()[0]


# ----------- standalone test code -----------
# if __name__ == "__main__":
#     import numpy as np
#     print("Starting VGG19 Test...")

#     # fake OpenCV image (100x100 pixel black square)
#     dummy_image = np.zeros((100, 100, 3), dtype=np.uint8) 

#     # Run through model
#     print("Processing image...")
#     features = extract_features(dummy_image)

#     print(f"Success! Extracted a list of {len(features)} numbers.")
#     print(f"First 5 numbers in the fingerprint: {features[:5]}")