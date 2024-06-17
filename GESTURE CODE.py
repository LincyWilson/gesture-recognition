# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 15:04:44 2023

@author: CC
"""


import cv2
import os
import time

root_dir = "gesture_data"
if not os.path.exists(root_dir):
    os.mkdir(root_dir)

train_dir = os.path.join(root_dir, "train")
if not os.path.exists(train_dir):
    os.mkdir(train_dir)

valid_dir = os.path.join(root_dir, "valid")
if not os.path.exists(valid_dir):
    os.mkdir(valid_dir)

gestures = ["pause", "play", "volume_up","volume_down" ,"next","previous"]
for gesture in gestures:
    train_gesture_dir = os.path.join(train_dir, gesture)
    if not os.path.exists(train_gesture_dir):
        os.mkdir(train_gesture_dir)

    valid_gesture_dir = os.path.join(valid_dir, gesture)
    if not os.path.exists(valid_gesture_dir):
        os.mkdir(valid_gesture_dir)

cap = cv2.VideoCapture(0)
gesture = input("Enter gesture name: ")
is_train = input("Enter 't' for train directory and 'v' for valid directory: ").lower() == 't'
gesture_dir = os.path.join(train_dir if is_train else valid_dir, gesture)
if not os.path.exists(gesture_dir):
    os.mkdir(gesture_dir)
num_photos = int(input("Enter number of images:"))

count = 0
start_time = None
while count < num_photos:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = frame[100:300, 100:300]
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
    
    if start_time is None:
        cv2.putText(frame, f"Press 'k' to start capturing images", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        seconds_left = 2 - int(time.time() - start_time)
        if seconds_left < 0:
            seconds_left = 0
        cv2.putText(frame, f"Captured {count} images, {seconds_left}s left", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Live Feed", frame)
    cv2.imshow("Region of Interest", roi)
   
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('k'):
        start_time = time.time()
   
    if start_time is not None and time.time() - start_time >= 2:
        filename = os.path.join(gesture_dir, f"{gesture}_{count}.jpg")
        cv2.imwrite(filename, roi)
        count += 1
        start_time = time.time()

cap.release()
cv2.destroyAllWindows()




import cv2
import os

# define path 
directory = r"C:\Users\CC\gesture_data\train\pause"

#Loop through all files in the directory 
for filename in os.listdir(directory):
    #load the image
    img = cv2.imread(os.path.join(directory, filename))
    print(img)

#convert image to greyscale 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Apply sobel edge detection 
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0 , ksize = 5)
sobely= cv2.Sobel(gray, cv2.CV_64F, 0,1, ksize= 5)
edges = cv2.addWeighted(sobelx, 0.5, sobely,0.5, 0)


#saving the image 
cv2.imwrite(os.path.join(directory, 'edges_' + filename), edges)


from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

cudnn.benchmark = True
plt.ion()   # interactive mode


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = r"C:/Users/CC/gesture_data"
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'valid']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])


def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['valid']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
        
        
model_ft = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
model_ft.fc = nn.Linear(num_ftrs,6 )

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=10)

visualize_model(model_ft)