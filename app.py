from flask import Flask, request, render_template, redirect, url_for
import onnxruntime as ort
import numpy as np
from PIL import Image
import pickle
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
MODEL_FOLDER = 'static/models'
ENCODER_FOLDER = 'static/encoders'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['ENCODER_FOLDER'] = ENCODER_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)
if not os.path.exists(ENCODER_FOLDER):
    os.makedirs(ENCODER_FOLDER)

history = []
session = None
deployed_encoder = None
deployed_model = None  # Track the currently deployed model
encoder = None  # Store the loaded encoder
current_file = None  # Track the current uploaded file


def load_model(model_path):
    """Load the ONNX model dynamically."""
    global session, deployed_model
    session = ort.InferenceSession(model_path)
    deployed_model = os.path.basename(model_path)
    print(f"Model loaded: {model_path}")


def load_encoder(encoder_path):
    """Load the encoder dynamically."""
    global deployed_encoder, encoder
    with open(encoder_path, 'rb') as file:
        encoder = pickle.load(file)  # Deserialize the encoder
    deployed_encoder = os.path.basename(encoder_path)
    print(f"Encoder loaded: {encoder_path}")


@app.route('/', methods=['GET', 'POST'])
def home():
    global session, history, deployed_encoder, deployed_model, encoder, current_file
    prediction = None
    confidence = None
    uploaded_image_url = None

    if request.method == 'POST':
        # Handle model upload
        if 'modelfile' in request.files and request.files['modelfile'].filename.endswith('.onnx'):
            model_file = request.files['modelfile']
            model_name = secure_filename(model_file.filename)
            model_path = os.path.join(app.config['MODEL_FOLDER'], model_name)
            model_file.save(model_path)
            load_model(model_path)  # Dynamically load the new model
            return render_template('index.html', deployed_model=deployed_model, history=history)

        # Handle encoder upload
        if 'encoderfile' in request.files and request.files['encoderfile'].filename.endswith('.pkl'):
            encoder_file = request.files['encoderfile']
            encoder_name = secure_filename(encoder_file.filename)
            encoder_path = os.path.join(app.config['ENCODER_FOLDER'], encoder_name)
            encoder_file.save(encoder_path)
            load_encoder(encoder_path)  # Dynamically load the new encoder
            return render_template('index.html', deployed_encoder=deployed_encoder, history=history)

        # Handle image upload
        if 'imagefile' in request.files:
            image_file = request.files['imagefile']
            file_name = secure_filename(image_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
            image_file.save(file_path)
            current_file = {"file_name": file_name, "image_url": url_for('static', filename=f'uploads/{file_name}')}
            return render_template('index.html', deployed_model=deployed_model, history=history, current_file=current_file)

        # Handle prediction
        if session and encoder and current_file:
            try:
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], current_file['file_name'])
                image = Image.open(image_path)
                image_array = np.array(image.convert('L').resize((64, 64))).astype(np.float32) / 255.0
                image_array = np.expand_dims(image_array, axis=0)
                image_array = np.expand_dims(image_array, axis=-1)

                # Predict using the current model
                input_name = session.get_inputs()[0].name
                output_name = session.get_outputs()[0].name
                predictions = session.run([output_name], {input_name: image_array})
                predicted_class_index = np.argmax(predictions[0])
                predicted_label = encoder.inverse_transform([predicted_class_index])
                prediction = predicted_label[0]
                confidence = predictions[0][0][predicted_class_index] * 100

                # Add to history
                history.append({
                    "prediction": prediction,
                    "confidence": f'{confidence:.1f}',
                    "file_name": current_file["file_name"],
                    "image_url": current_file["image_url"]
                })

                return render_template(
                    'index.html',
                    prediction=prediction,
                    confidence=f'{confidence:.1f}',
                    deployed_model=deployed_model,
                    history=history,
                    current_file=current_file
                )

            except Exception as e:
                return f"Error processing file: {e}", 500

    return render_template('index.html', deployed_model=deployed_model, history=history, current_file=current_file)


@app.route('/clear_history', methods=['POST'])
def clear_history():
    global history
    history = []
    return redirect(url_for('home'))


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))  # Use PORT env variable or default to 3000
    app.run(host='0.0.0.0', port=port, debug=True)
