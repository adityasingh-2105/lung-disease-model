# Lung Disease Classifier

A simple and lightweight Convolutional Neural Network (CNN) model for classifying chest X-ray images into **Normal** and **Pneumonia** categories. The model is deployed as a clean and minimal Streamlit web application.

## Features

- **Lightweight CNN Architecture**: Simple 3-layer convolutional network with max pooling
- **Easy to Train**: Straightforward training script with progress tracking
- **User-Friendly Web App**: Clean Streamlit interface for image upload and prediction
- **Minimal Dependencies**: Uses only essential libraries (PyTorch, Streamlit, PIL)

## Project Structure

```
Lung Disease IMD/
├── model.py           # CNN model architecture
├── preprocessing.py   # Data preprocessing utilities
├── train.py          # Training script
├── app.py            # Streamlit web application
├── requirements.txt  # Python dependencies
├── README.md         # Project documentation
└── models/           # Directory for saved models (created after training)
```

## Installation

1. **Clone or download this repository**

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation

Organize your dataset in the following structure:

```
data/
├── train/
│   ├── Normal/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── Pneumonia/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── val/
    ├── Normal/
    │   ├── image1.jpg
    │   └── ...
    └── Pneumonia/
        ├── image1.jpg
        └── ...
```

**Note:** The dataset directories should contain subdirectories named after the classes (e.g., "Normal", "Pneumonia").

## Training the Model

Train the model using the provided training script:

```bash
python train.py --data_dir data --epochs 10 --batch_size 32 --lr 0.001
```

**Arguments:**
- `--data_dir`: Path to dataset directory (should contain `train/` and `val/` subdirectories)
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size for training (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--model_dir`: Directory to save trained models (default: `models`)
- `--num_classes`: Number of classes (default: 2)
- `--input_channels`: Number of input channels - 1 for grayscale, 3 for RGB (default: 1)

**Example:**
```bash
python train.py --data_dir ./chest_xray --epochs 20 --batch_size 64
```

The trained model will be saved as `models/best_model.pth` based on validation accuracy.

## Running the Web Application

1. **Make sure you have a trained model** in the `models/` directory (`best_model.pth`)

2. **Launch the Streamlit app**
   ```bash
   streamlit run app.py
   ```

3. **Open your browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

4. **Upload an X-ray image** and view the prediction results!

## Model Architecture

The CNN uses a simple architecture:

- **3 Convolutional Blocks**: Each with Conv2D + ReLU + MaxPooling
  - Block 1: 32 filters
  - Block 2: 64 filters
  - Block 3: 128 filters
- **2 Fully Connected Layers**: 512 hidden units + output layer
- **Dropout**: 0.5 for regularization
- **Input Size**: 224x224 grayscale images
- **Output**: 2 classes (Normal, Pneumonia)

## Usage Example

```python
from model import create_model
from preprocessing import preprocess_image
import torch

# Load model
model = create_model(pretrained_path='models/best_model.pth')

# Preprocess image
image_tensor = preprocess_image('path/to/xray.jpg')

# Make prediction
with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    predicted_class = probabilities.argmax().item()
    
print(f"Predicted class: {['Normal', 'Pneumonia'][predicted_class]}")
```

## Important Notes

⚠️ **Medical Disclaimer**: This tool is for educational and demonstration purposes only. It should **NOT** be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns.

## Requirements

- Python 3.7+
- PyTorch 2.0+
- Streamlit 1.28+
- Pillow 9.0+
- NumPy 1.24+

## Troubleshooting

- **Model not found error**: Make sure you've trained the model first and it's saved in `models/best_model.pth`
- **CUDA out of memory**: Reduce the batch size in training or use CPU mode
- **Image format errors**: Ensure uploaded images are in PNG, JPG, or JPEG format

## License

This project is for educational purposes.

## Acknowledgments

This project demonstrates basic CNN implementation for medical image classification using PyTorch and Streamlit.

