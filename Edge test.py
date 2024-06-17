# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 16:48:05 2023

@author: CC
"""

import cv2
import os

# define path 
directory = 'C:/Users/CC/gesture_data/valid/volume_up'

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