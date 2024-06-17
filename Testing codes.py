# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 13:11:33 2023

@author: CC
"""

import torch
import torchvision.transforms as transforms
import cv2
from PIL import Image
# importing time for setting timer
import time

# Load the saved model
# model = torch.load('model_quantized_and_trained.pth')

model = torch.jit.load('resnet_18_model2.pt')


# Define the image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Define the class labels
label_dict = {0: 'next', 1: 'pause', 2: 'play',3:'previous',4:'volume_down',5:'volume_up' }

# Set up the webcam
cap = cv2.VideoCapture(0)
start_time = None

# Loop over frames from the webcam
while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    roi = frame[100:300, 100:300]
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
    #if start_time is None:
     #   cv2.putText(frame, f"Press 's' to start capturing images", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #if not ret:
     #   break

    # Convert the frame to an image and apply the image transform
    image = Image.fromarray(frame)
    image = transform(image)
    image = image.unsqueeze(0)

    # Make a prediction using the trained model
    model.eval()
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output,dim = 1)
    _, predicted = torch.max(output, 1)
    predicted_label = predicted.item()
    predicted_name = label_dict[predicted_label]
    prob = probs[0][predicted].item()

    # Display the predicted gesture name on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 255, 0)
    thickness = 2
    x = 10
    y = 50
    cv2.putText(frame, f'{predicted_name}: {prob:.2f}', (10, 30), font, font_scale, color, thickness)

    # Show the frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()