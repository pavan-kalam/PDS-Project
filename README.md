# Deepfake

This project implements a deep learning model for detecting deepfake images using TensorFlow and Keras. The model is trained to classify images as either real or fake, utilizing a dataset of fake and real images.

## Project Implementation Description

In view of our limited local computational resources, and to make the model training faster, we utilized Google Colab to implement this project. (By running all the cells in the colab)

Also, the model can be created using local system with limited computational resource by running 'python app.py' in the terminal, but it will take more time to train the model. (By running app.py)

## Download dataset
The dataset was directly downloaded from Kaggle to the project by the below kaggle credentials,

Kaggle username: [pavankalam]
Kaggle key: [23347953]



## Table of Contents
- [Installation] 
- [Loading-the-dataset]
- [Loading-the-model-from-colab-to-local-system-project0directory]
- [Usage]


## Installation

To run this project, you will need to install the required libraries in a new environment, which are available in 'requirements.txt', by running the following command:

```shell
python -m venv deepfake_env
# Activate the environment
# On Windows
deepfake_env\Scripts\activate
# On macOS/Linux
source deepfake_env/bin/activate
# Install requirements
pip install -r requirements.txt
```

## Loading the dataset

Loading the dataset from kaggle to the project either in colab or in local system, by using above mention Kaggle credentials.


## Loading the model from colab to local system project directory

As the model is training in the colab , we saved the model in the colab mounted google drive, and the same model is loaded to the local system project directory manually.

## Usage

here using the model to predict the images, by running the below script in the local system project directory,

```shell
python app_interface.py
```

This will start the application interface, where we can upload the the images and predict the results (real or fake)

Application interface URL: http://127.0.0.1:5000



