import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import os
import matplotlib.pyplot as plt
import opendatasets as od
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

model_path = 'model/test/cnn_original.keras'
model_name = 'cnn_original'
# Download dataset function
def download_dataset():
    if not os.path.exists('raw_data/deepfake-and-real-images'):
        print("Downloading dataset...")
        od.download("https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images")
        print("Dataset downloaded successfully!")
    else:
        print("Dataset already exists!")

# Model creation function
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation="relu", padding='same', input_shape=(128, 128, 3)),
        layers.Conv2D(32, (3,3), activation="relu", padding='same'),
        layers.MaxPooling2D((2,2), strides=(2, 2)),

        layers.Conv2D(64, (3,3), activation="relu", padding='same'),
        layers.Conv2D(64, (3,3), activation="relu", padding='same'),
        layers.MaxPooling2D((2,2), strides=(2, 2)),

        layers.Conv2D(128, (3,3), activation="relu", padding='same'),
        layers.Conv2D(128, (3,3), activation="relu", padding='same'),
        layers.MaxPooling2D((2,2), strides=(2, 2)),

        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    
    return model

# Function to load and preprocess the dataset
def load_dataset(path):
    return tf.keras.utils.image_dataset_from_directory(
        path,
        labels='inferred',
        label_mode='binary',
        color_mode="rgb",
        batch_size=32,
        image_size=(128, 128),
        shuffle=True,
        seed=123,
    )

# Function to plot training history
def plot_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('visualization/training_history.png')

# Function to create and save confusion matrix
def create_confusion_matrix(model, val_data, model_name):
    y_true = []
    y_pred = []

    for images, labels in val_data:
        predictions = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend((predictions > 0.5).astype(int).flatten())

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["REAL", "FAKE"])
    
    plt.figure(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {model_name}')
    
    # Save the confusion matrix with the model name in the filename
    plt.savefig(f'visualization/confusion_matrix_{model_name}.png')
    plt.close()

# Main execution
if __name__ == '__main__':
    print("Checking for dataset...")
    download_dataset()
    
    # Create necessary directories
    if not os.path.exists('model'):
        os.makedirs('model')
    
    # Load and preprocess the dataset
    dataset_path = 'raw_data/deepfake-and-real-images/Dataset'
    train_data = load_dataset(os.path.join(dataset_path, 'Train'))
    val_data = load_dataset(os.path.join(dataset_path, 'Validation'))
    
    
    if os.path.exists(model_path):
        print("Loading existing model...")
        model = keras.models.load_model(model_path)
    else:
        print("Creating and training a new model...")
        model = create_model()
        history = model.fit(train_data, epochs=20, validation_data=val_data)
        model.save(model_path)  # Save the trained model
        plot_history(history)  # Plot and save training history
        print("Model trained and saved successfully.")

    # Create and save confusion matrix with model name
    create_confusion_matrix(model, val_data, model_name)
    print("Confusion matrix created and saved successfully.")