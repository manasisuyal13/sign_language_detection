---

# Sign Language Detection System

## Overview

The **Sign Language Detection System** is a machine learning-based project that can recognize sign language gestures in real time using a webcam or an uploaded image. The system utilizes a Convolutional Neural Network (CNN) model trained to classify hand gestures representing letters from the American Sign Language (ASL) alphabet.

This project aims to provide an accessible way for computers to understand and interpret sign language, facilitating communication for individuals with hearing or speech impairments.

---

## Key Features

- **Real-time Gesture Recognition**: Detects ASL alphabet signs and common phrases using a webcam feed.
- **Image-based Gesture Prediction**: Users can upload an image, and the model will predict the corresponding hand gesture.
- **Webcam Support**: The system can process video input from the webcam to detect and predict ASL gestures in real time.
- **CNN Model**: A trained CNN model that classifies hand gestures based on images.

---

## Technologies Used

- **Python**: The main programming language.
- **PyTorch**: For model building and training.
- **OpenCV**: For real-time webcam input processing.
- **Pillow (PIL)**: For image handling and transformations.
- **NumPy**: For data manipulation and calculations.
- **TorchVision**: For dataset management and data augmentation.

---

## Prerequisites

Before running the project, ensure you have the following:

- Python 3.7+
- Pip package manager

Install required Python libraries using the following command:

```bash
pip install -r requirements.txt
```

---

## Steps to Run the Project

### Step 1: Clone the Repository

Clone this repository to your local machine using Git:

```bash
git clone https://github.com/username/sign-language-detection.git
cd sign-language-detection
```

### Step 2: Dataset Setup

To train the model, you need to download the **ASL Alphabet Dataset**. This dataset contains images of hand gestures corresponding to each letter of the American Sign Language alphabet.

1. Download the **ASL Alphabet dataset** from [this Kaggle link](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).
2. Extract the dataset and place it in the following directory structure:

```
archive/asl_alphabet_train/
    └── A/
    └── B/
    └── C/
    └── ...
```

Each folder corresponds to a letter of the ASL alphabet and contains images of that letter's hand gesture.

### Step 3: Train the Model

To train the CNN model, follow these steps:

1. Ensure the dataset is set up as shown in the previous step.
2. Run the `train_cnn.py` script to start the training process:

```bash
python train_cnn.py
```

This script will:
- Preprocess the dataset (resize images, normalize pixel values).
- Train a Convolutional Neural Network (CNN) on the dataset.
- Save the trained model to a file called `asl_cnn_model.pth` once training is complete.

### Step 4: Making Predictions

Once the model is trained, you can use it to make predictions either from an uploaded image or through a live webcam feed.

#### Option 1: **Image-based Prediction**

To make predictions from an image, run the following:

```bash
python predict_image.py --image_path "path_to_image.jpg"
```

This will load the image, preprocess it, and predict the hand gesture corresponding to the ASL sign.

#### Option 2: **Webcam-based Prediction**

For real-time predictions via webcam, run:

```bash
python webcam_input.py
```

This will start the webcam and continuously display predictions on the screen as hand gestures are detected.

### Step 5: Using the Trained Model

If you'd like to use the trained model directly, you can load it using PyTorch and make predictions. Here's an example script for making predictions from an image:

```python
import torch
from PIL import Image
from torchvision import transforms
from model import ASLCNN  # Ensure your model class is in model.py

# Load the trained model
model = ASLCNN(num_classes=29)
model.load_state_dict(torch.load("asl_cnn_model.pth"))
model.eval()

# Image Transformation
transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

# Load and preprocess the image
image_path = "test_image.jpg"
image = Image.open(image_path).convert('RGB')
image = transform(image).unsqueeze(0)

# Make the prediction
output = model(image)
predicted_class = torch.argmax(output, dim=1)

# Print the predicted class (index of ASL gesture)
print(f"Predicted Gesture: {predicted_class.item()}")
```

---

## CNN Model Architecture

The CNN model used in this project consists of three convolutional layers followed by fully connected layers for classification. Here's a breakdown of the architecture:

1. **Convolutional Layers**:
   - The first three layers use convolution to capture spatial features from the hand gestures.
   - Each convolutional layer is followed by a ReLU activation function and a max-pooling layer to reduce the spatial dimensions.

2. **Fully Connected Layers**:
   - The features from the convolutional layers are flattened and passed through two fully connected layers.
   - A dropout layer is applied to prevent overfitting.

---

## Project Structure

Here is the folder structure of the project:

```
sign-language-detection/
    ├── archive/
    │   └── asl_alphabet_train/
    ├── model.py                # Contains the CNN model class
    ├── train_cnn.py            # Script to train the CNN model
    ├── predict_image.py        # Script to make predictions from an image
    ├── webcam_input.py         # Script to make predictions from the webcam
    ├── requirements.txt        # List of dependencies
    └── README.md               # Project documentation
```

---

## Model Training Details

### Data Preprocessing:
- **Image Resizing**: Each image is resized to a standard 64x64 pixels.
- **Normalization**: Pixel values are normalized to range from 0 to 1 to improve model convergence during training.

### CNN Model Architecture:

```python
class ASLCNN(nn.Module):
    def __init__(self, num_classes):
        super(ASLCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
```

### Training Loop:
- **Optimizer**: Adam optimizer is used for training the model.
- **Loss Function**: Cross-entropy loss is used for multi-class classification.

### Saving the Model:
After training, the model is saved as `asl_cnn_model.pth`:

```python
torch.save(model.state_dict(), "asl_cnn_model.pth")
```

---

## Conclusion

This **Sign Language Detection System** can be used in a variety of applications, including helping people with hearing or speech impairments communicate more easily with technology. The model is trained to recognize the ASL alphabet and can be extended to support more gestures or other sign languages.

Feel free to fork this repository, report issues, and contribute improvements to the project.
