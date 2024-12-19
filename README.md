# Musical Note Classifier 🎵

This project uses a deep learning model to classify musical notes based on their pitch and length. The Flask application allows users to deploy their own models and perform predictions interactively.

## Features
- Upload `.onnx` models for deployment.
- Classify uploaded images of musical notes.
- View prediction history with confidence scores.

## Hosted Application
The live application is hosted on Render:
[Musical Note Classifier](https://musical-note-classifier.onrender.com)

Note: if you want to run the model on the website, you will need to host the website locally. Instructions below.

## Download Files
- [Download Model (cnn-musical-note-classifier.onnx)](https://github.com/dongim04/musical-note-classifier/raw/main/cnn-musical-note-classifier.onnx)
- [Download Encoder (encoder.pkl)](https://github.com/dongim04/musical-note-classifier/raw/main/encoder.pkl)

## Run Locally
1. Clone this repository:
   ```bash
   git clone https://github.com/dongim04/musical-note-classifier.git
   cd musical-note-classifier
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the Flask application:
   ```bash
   python app.py
   ```
4. Open `http://127.0.0.1:3000` in your browser.
