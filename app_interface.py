import os
import io
import numpy as np
from flask import Flask, render_template, request
from PIL import Image
from tensorflow import keras

app = Flask(__name__)

# Global variable for the model and history
model = None
history = []  # Global list to store history of predictions
# Define a directory for uploaded images
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load your model here
model = keras.models.load_model('model/cnn_original.keras')  # Adjust the filename as necessary

def preprocess_image(image):
    # Ensure the image is in RGB format
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize the image to the expected input size for the model
    image = image.resize((128, 128))  # Adjust size based on your model's input shape
    
    # Convert the image to a numpy array and normalize pixel values
    image_array = np.array(image)
    image_array = image_array.astype('float32') / 255.0  # Normalize to [0, 1]
    
    # Expand dimensions to match model input shape (1, height, width, channels)
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file uploaded')
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', error='No file selected')
    
    try:
        # Save the uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Reset the file pointer to the beginning
        file.seek(0)

        # Process the image
        image = Image.open(file)  # Open the image directly from the file object
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image)[0][0]
        
        result = "FAKE" if prediction >= 0.5 else "REAL"
        confidence = float(prediction) if prediction >= 0.5 else float(1 - prediction)
        confidence = confidence * 100
        
        # Append the result to history
        history.append({
            'filename': file.filename,
            'result': result,
            'confidence': "%.2f" % confidence
        })

        return render_template('results.html', 
                               result=result, 
                               confidence=confidence,
                               filename=file.filename)
    
    except Exception as e:
        return render_template('index.html', error=f'Error: {str(e)}')

@app.route('/history')
def history_page():
    global history  # Ensure you declare it as global here too
    return render_template('history.html', history=history)

if __name__ == '__main__':
    app.run(debug=True)