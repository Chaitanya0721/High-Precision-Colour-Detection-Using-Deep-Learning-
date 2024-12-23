# High-Precision-Colour-Detection-Using-Deep-Learning-CNN

Here’s a README template for your GitHub repository based on the project you've described:

```markdown
# Color Classification CNN

This is a deep learning project that uses Convolutional Neural Networks (CNN) to classify images based on their high-intensity colors. The model is trained on a dataset containing 8 color classes: black, blue, green, orange, red, violet, white, and yellow. The model predicts the color of an object in a given image.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview
This project uses a CNN to classify images into one of eight color categories. The model was trained on a dataset of images and is capable of recognizing the dominant color in new images. The model is built using TensorFlow/Keras and can be run on local machines or Google Colab.

## Installation
To run this project, you will need to install the required dependencies. You can do this using pip:

```bash
pip install tensorflow numpy matplotlib pandas
```

## Dataset
The dataset used in this project consists of images belonging to 8 color classes: black, blue, green, orange, red, violet, white, and yellow. The dataset is organized as follows:

```
/content/drive/MyDrive/DL Datasets/
    ├── training_dataset/
        ├── black/
        ├── blue/
        ├── green/
        ├── orange/
        ├── red/
        ├── violet/
        ├── white/
        ├── yellow/
    ├── validation_dataset/
        ├── black/
        ├── blue/
        ├── green/
        ├── orange/
        ├── red/
        ├── violet/
        ├── white/
        ├── yellow/
```

### Image Dimensions
- **Target size**: 64x64 pixels
- **Number of classes**: 8 (representing the color categories)

## Model Architecture
The model consists of the following layers:
- **Conv2D layers**: 3 convolutional layers with 32, 64, and 128 filters, respectively.
- **MaxPooling2D layers**: After each convolutional layer, max-pooling is applied to reduce the spatial dimensions.
- **Dense Layer**: A fully connected layer with 256 neurons.
- **Dropout Layer**: Dropout is used for regularization to prevent overfitting.
- **Output Layer**: 8 neurons (for 8 color classes) with a softmax activation function.

### Model Summary
```bash
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 62, 62, 32)        896       
                                                                 
 max_pooling2d (MaxPooling2  (None, 31, 31, 32)        0         
 D)                                                              
                                                                 
 conv2d_1 (Conv2D)           (None, 29, 29, 64)        18496     
                                                                 
 max_pooling2d_1 (MaxPoolin  (None, 14, 14, 64)        0         
 g2D)                                                            
 ... (other layers) ...                                          
=================================================================
Total params: 1,275,208 (4.86 MB)
Trainable params: 1,275,208 (4.86 MB)
Non-trainable params: 0
_________________________________________________________________
```

## Training
The model was trained using the following configuration:
- **Optimizer**: Adam
- **Loss function**: Categorical Cross-Entropy
- **Metrics**: Accuracy
- **Data Augmentation**: Random transformations like shear, zoom, and horizontal flip to prevent overfitting.
- **Class Weights**: Calculated to address class imbalance during training.

The model was trained for 10 epochs, and accuracy improved steadily during training.

## Usage

To use the trained model for predicting the color of a new image, use the following function:

```python
def predict_color(image_path):
    img = load_and_preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_color_index = np.argmax(prediction)
    predicted_color = color_labels[predicted_color_index]
    return predicted_color

def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array
```

### Example Usage
```python
test_image_path = '/path/to/test_image.jpg'
predicted_color = predict_color(test_image_path)
print(f"Predicted Color: {predicted_color}")
```

## Contributing
Contributions are welcome! If you want to contribute to this project, please fork the repository, create a new branch, and submit a pull request. Ensure your code is properly tested and adheres to the existing code style.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

---

This README covers the core sections of the project, providing clear instructions on how to set up, use, and contribute to the repository. You can further customize the content based on your project's needs.
