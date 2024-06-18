# Gesture Recognition using Convolutional Neural Networks


## Overview

This repository contains the implementation of a gesture recognition system using Convolutional Neural Networks (CNNs). The system is designed to recognize hand gestures and map them to music control actions such as play, pause, skip, and volume control.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Training](#model-training)
- [Results](#results)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)

## Introduction

Gesture recognition enables intuitive and natural human-computer interactions by identifying and interpreting human gestures. This technology has applications in various domains including virtual reality, gaming, healthcare, and robotics.

The primary goal of this project is to develop a gesture recognition system to control music playback using hand gestures. The system recognizes gestures such as play, pause, volume up, volume down, next, and previous, and maps them to corresponding music control actions.

## Dataset

The dataset consists of images representing six different hand gestures:

- Play
- Pause
- Volume Up
- Volume Down
- Next
- Previous

### Data Collection

- **Create Directories**: The dataset is organized into train and valid directories, each containing subdirectories for the six gestures.
- **Capture Images**: The script captures grayscale images from the camera, focusing on a region of interest (ROI) between coordinates (100, 100) and (300, 300).

### Preprocessing

- **Edge Detection**: Apply Sobel edge detection to highlight the edges in the images.
- **Data Augmentation**: Transformations such as resizing, random cropping, and horizontal flipping are applied to enhance the dataset.

## Methodology

### Model Architecture

We utilized the ResNet-18 architecture pre-trained on the ImageNet dataset. The final layer of the network was reinitialized to output six classes corresponding to the gestures.

### Training

- **Data Loaders**: Prepare PyTorch data loaders for training and validation datasets.
- **Training Process**: Train the model using the AdamW and SGD optimizers with different configurations.
- **Evaluation**: Validate the model after each epoch and save the best-performing model.

## Model Training

### Steps

1. **Define Data Transforms**: Apply appropriate transformations for training and validation sets.
2. **Prepare Datasets and Data Loaders**: Use ImageFolder and DataLoader to load and iterate over the dataset.
3. **Train Model**: Implement the training loop with loss computation, backpropagation, and optimizer steps.
4. **Visualize Results**: Display the training progress and visualize the model predictions on validation data.

### Fine-Tuning and Feature Extraction

Two approaches were explored:

- **Fine-Tuning**: Adjusting all model parameters based on the new task.
- **Fixed Feature Extractor**: Only the final layer is trained while keeping the pre-trained layers frozen.

## Results

| Model                        | Training Loss | Validation Loss | Validation Accuracy |
|------------------------------|---------------|-----------------|---------------------|
| Fine-Tuned ResNet_18_model5  | 0.1918        | 0.2895          | 94.44%              |
| Fixed Feature ResNet_18_model6 | 0.1902        | 0.3082          | 91.50%              |
| Fine-Tuned ResNet_18_model1  | 0.0458        | 0.0474          | 99.02%              |
| Fixed Feature ResNet_18_model2 | 0.2733        | 0.2309          | 97.71%              |

## Future Work

- **Advanced Algorithms**: Explore transformers, reinforcement learning, and GANs for improved performance.
- **Integration**: Combine with NLP and other computer vision techniques for enhanced capabilities.
- **Scalability**: Utilize cloud platforms for handling larger datasets and more complex models.
- **Applications**: Implement in chatbots, virtual assistants, and recommendation systems.
- **User Experience**: Develop intuitive web or mobile applications for better user interaction.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

Special thanks to the open-source community and the authors of the pre-trained ResNet model. This project was made possible by the contributions and support of various developers and researchers in the field of deep learning.
