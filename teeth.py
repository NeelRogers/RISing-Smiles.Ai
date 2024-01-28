import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os
from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
import base64
from io import BytesIO
import webview
from threading import Thread

app = Flask(__name__)

# Load the pre-trained ResNet-18 model
model_path = 'teeth_type_Detection.pth'
new_model = models.resnet18(pretrained=True)
new_model.fc = torch.nn.Linear(new_model.fc.in_features, 4)  # Assuming 4 classes
new_model.eval()

def run_flask():
    os.system("teeth.py")

if __name__ == '__main__':
    t = Thread(target=run_flask)
    t.start()

    webview.create_window("RISing Smile App", "http://127.0.0.1:29928/")
    webview.start()

def get_class_label(class_index):
    class_names = ['cavity', 'crocked teeth', 'edgeofcrown', 'plaque']
    return class_names[class_index]

def process_image(img_data):
    # Decode base64 and convert to NumPy array
    img_data = base64.b64decode(img_data.split(',')[1])
    img_np = np.array(Image.open(BytesIO(img_data)))

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(Image.fromarray(img_np))
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = new_model(input_batch)

    _, predicted_class = torch.max(output, 1)
    return predicted_class.item()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image_route():
    img_data = request.form['image_data']
    predicted_class = process_image(img_data)
    return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    webview.create_window("RISing Smile App", app, width=800, height=600)
    webview.start(debug=True)

# Define data transformations for data augmentation and normalization
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Define the data directory
data_dir = 'dataset'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
print(dataset_sizes)

class_names = image_datasets['train'].classes

# Load the pre-trained ResNet-18 model
resnet18 = models.resnet18(pretrained=True)

# Freeze all layers except the final classification Layer
for name, param in resnet18.named_parameters():
    if "fc" in name:  # Unfreeze the final classification Layer
        param.requires_grad = True
    else:
        param.requires_grad = False

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet18.parameters(), lr=0.001, momentum=0.9)  # Use all parameters

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18.to(device)

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

print("Training complete!")
model_path = 'teeth_type_Detection.pth'
torch.save(resnet18.state_dict(), model_path)

# Inference
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1000)
model.load_state_dict(torch.load('teeth_type_Detection.pth'))
model.eval()

new_model = models.resnet18(pretrained=True)
new_model.fc = nn.Linear(new_model.fc.in_features, 2)

new_model.fc.weight.data = model.fc.weight.data[0:2]
new_model.fc.bias.data = model.fc.bias.data[0:2]

# Process and classify images
image_path = 'C:\\Users\\admin\\Desktop\\teeth_tales_app\\dataset\\train\\cavity'
image_files = glob.glob(image_path + '/*.jpg')
for model_path in image_files:
    image = Image.open(model_path)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = new_model(input_batch)

    value = [0]
    _, predicted_class = output, max(value)

    class_names = ['cavity', 'crocked teeth', 'edgeofcrown', 'plaque']
    predicted_class_name = class_names[predicted_class]

    print(f'The predicted class for {model_path} is: {predicted_class_name}')

    # Display the image with the predicted class
    image = np.array(Image)
    plt.imshow(image)
    plt.axis('off')
    plt.text(10, 10, f'Predicted: {predicted_class_name}', fontsize=17, color="white", backgroundcolor="black")
    plt.show()
