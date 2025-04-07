from flask import Flask, request, render_template, send_from_directory, redirect, url_for, session
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import confusion_matrix
from PIL import Image
import torch.nn.functional as F

app = Flask(__name__)
app.secret_key = "\xf0?a\x9a\\\xff\xd4;\x0c\xcbHi"

matplotlib.use('agg')

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max file size: 16 MB

# Model 1
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        # MaxPooling layer
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Model 2
class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Model 3
class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> 32x14x14
        x = self.pool(F.relu(self.conv2(x)))  # -> 64x7x7
        x = self.pool(F.relu(self.conv3(x)))  # -> 128x3x3
        x = x.view(-1, 128 * 3 * 3)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
@app.route('/index')
def index():
    cm_model_path = request.args.get('cm_model_path')
    training_loss = request.args.get('training_loss')
    model_name = request.args.get('model_name')

    return render_template('index.html', cm_model_path=cm_model_path, training_loss=training_loss, model_name=model_name)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    if file and allowed_file(file.filename):
        select = request.form.get('true-values')
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        model_type = session.get('model-type')

        # Load and preprocess the image
        image_path = file_path
        image = Image.open(image_path).convert('L')
        image = image.resize((28, 28))
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)

        # Load the pre-trained model (assuming you have one)
        if model_type == "Model 1":
            model = CNN()
            model.load_state_dict(torch.load('model.pth'))
        elif model_type == "Model 2":
            model = FNN()
            model.load_state_dict(torch.load('model2.pth'))
        elif model_type == "Model 3":
           model = DeepCNN()
           model.load_state_dict(torch.load('model3.pth')) 
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.eval()

        # Make predictions
        with torch.no_grad():
            output = model(image_tensor)
            probs = F.softmax(output, dim=1)  
            confidence, predicted = torch.max(probs, 1)

        prediction = predicted.item()
        confidence_percent = round(confidence.item() * 100, 2)
        
        # Create Confusion Matrix
        cm = np.zeros((10, 10), dtype="int")
        cm[int(select), prediction] = 1

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix for User's Digit")

        cm_path = "uploads/confusion_matrix.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()

        return render_template('model.html', file_path=file_path, prediction=prediction, cm_path=cm_path, confidence_percent=confidence_percent)

    return "Invalid file type. Only images are allowed."

@app.route('/refresh', methods=['POST'])
def model_images():
    model_name = request.form.get('model-type')
    session['model-type'] = request.form.get('model-type')

    if model_name == "Model 1":
        cm_model_path = "static/images/model1_confusion_matrix.png"
        training_loss = "static/images/model1_trainingloss.png"
    elif model_name == "Model 2":
        cm_model_path = "static/images/model2_confusion_matrix.png"
        training_loss = "static/images/model2_trainingloss.png"
    elif model_name == "Model 3":
        cm_model_path = "static/images/model3_confusion_matrix.png"
        training_loss = "static/images/model3_trainingloss.png"

    return redirect(url_for('index', cm_model_path=cm_model_path, training_loss=training_loss, model_name=model_name))

# Route to serve the uploaded image
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/download")
def download_cm():
    return send_from_directory(app.config['UPLOAD_FOLDER'], "confusion_matrix.png")

if __name__ == "__main__":
    app.run(debug=True)
