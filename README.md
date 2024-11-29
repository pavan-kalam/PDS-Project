# Deepfake Detection Model

This project implements a deep learning model for detecting deepfake images using TensorFlow and Keras. The model is trained to classify images as either real or fake, utilizing a dataset of fake and real images.

## Project Implementation Description

In view of our limited local computational resources, we utilized Google Colab to implement this project.

The dataset was directly downloaded from Kaggle into Colab for ease of access.

Due to the storage requirements of our dataset and trained model, GitHub’s storage limitations prevented us from uploading them there. Therefore, we use the dataset directly from kaggle and model in Google Drive.

The respective file links are provided below.
- [Dataset] https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images
- [Model] https://drive.google.com/drive/folders/1A1of6h5H0SQeVI9jTi59uZlCCxc1oNRk?usp=sharing

## Table of Contents
- [Installation] 
- [Usage]
- [Model-Architecture]
- [Training]
- [Evaluation]
- [Image-Prediction]

## Installation

To run this project, you will need to install the required libraries. You can do this by running the following command:

```bash
pip install opendatasets tensorflow pillow matplotlib ipywidgets
```

## Usage

First, the dataset is automatically downloaded from Kaggle if it doesn't already exist in the Google Drive.

In second step, the model is trained on the dataset for a specified number of epochs.

After training, the model is evaluated using a test dataset, and a confusion matrix is displayed.

You can upload images to check if they are real or fake using the provided interface.

## Model-Architecture

The model consists of several convolutional layers followed by max pooling layers, a flatten layer, dropout for regularization, and dense layers for classification. 

The architecture is designed to effectively learn features from images to distinguish between real and fake content.

## Training

The model is trained using the Adam optimizer with a learning rate of 0.0001. 

Early stopping is implemented to prevent overfitting, and the model's performance is monitored using accuracy and loss metrics.

## Evaluation

After training, the model is evaluated on a test dataset, and a confusion matrix is generated to visualize the performance. 
The confusion matrix helps in understanding the number of true positives, true negatives, false positives, and false negatives.

## Image-Prediction

The model allows users to upload images for prediction. 

The uploaded image is preprocessed, and the model predicts whether the image is real or fake, along with a confidence score.