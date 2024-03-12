from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
model_path = "Resnet_fineTuning.pth"
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Define the classes
class_names = ['benign', 'malignant', 'normal']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file selected. Please choose an image."

    file = request.files['file']

    if file.filename == '':
        return "No file selected. Please choose an image."

    try:
        # Read the image file
        img = Image.open(io.BytesIO(file.read()))

        # Apply the transformation
        img_tensor = transform(img).unsqueeze(0)

        # Make the prediction
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)

        prediction_class = class_names[predicted.item()]

        # Specify detailed messages based on prediction
        if prediction_class == 'normal':
            result_message = ("The analysis shows a normal condition. Regular health check-ups and maintaining a "
                              "healthy lifestyle are recommended.")
        elif prediction_class == 'benign':
            result_message = ("Analysis suggests the presence of a benign condition. It is generally non-cancerous or "
                              "non-threatening. Regular check-ups are recommended for monitoring.")
        elif prediction_class == 'malignant':
            result_message = ("The analysis indicates the possibility of a malignant condition. It is crucial to "
                              "consult with a medical professional as soon as possible for a more comprehensive "
                              "evaluation.")

        return jsonify(dict(prediction=prediction_class, result_message=result_message))

    except Exception as e:
        return f"Error processing image: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
