# Facial Emotion Recognition (FER) System

A deep learning-based facial emotion recognition system that can detect and classify seven basic emotions from facial images: angry, disgust, fear, happy, neutral, sad, and surprise.

## Overview

This project implements a Convolutional Neural Network (CNN) for facial emotion recognition. The model is trained on grayscale facial images and can classify emotions with high accuracy. The system includes both a training notebook and a real-time emotion detection application.

## Features

- Real-time emotion detection from webcam feed
- Support for 7 basic emotions: angry, disgust, fear, happy, neutral, sad, and surprise
- Pre-trained model for immediate use
- Training notebook for model customization
- GPU acceleration support (optional)
- Mixed precision training for improved performance

## Project Structure

```
Facial-Emotions-Detection/
├── FER_model.ipynb          # Training notebook
├── app.py                   # Real-time emotion detection application
├── images/                  # Dataset directory
│   ├── train/              # Training images
│   └── test/               # Test images
└── requirements.txt         # Project dependencies
```

## Requirements

The following Python packages are required:

```
tensorflow
keras
pandas
numpy
jupyter
notebook
tqdm
opencv-contrib-python
scikit-learn
matplotlib
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Facial-Emotions-Detection.git
cd Facial-Emotions-Detection
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

1. Open `FER_model.ipynb` in Jupyter Notebook
2. Follow the notebook cells to:
   - Load and preprocess the dataset
   - Train the CNN model
   - Evaluate model performance
   - Save the trained model

### Real-time Emotion Detection

Run the application:
```bash
python app.py
```

The application will:
1. Open your webcam
2. Detect faces in real-time
3. Classify emotions
4. Display results with bounding boxes and emotion labels

## Model Architecture

The CNN model consists of:
- Multiple convolutional layers with ReLU activation
- MaxPooling layers for dimensionality reduction
- Dropout layers for regularization
- Dense layers for classification
- Softmax output layer for 7 emotion classes

## Performance

The model is trained on a large dataset of facial expressions and achieves high accuracy in emotion classification. Performance metrics are available in the training notebook.

## Contributing

Feel free to submit issues and enhancement requests!

## License

[Your chosen license]

## Acknowledgments

- Dataset: [Your dataset source]
- Inspired by: [Any inspirations or references]