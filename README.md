# Multilabel Classification with Various Base Models and Hyperparameter Tuning

This GitHub repository contains code for multilabel classification of 107 classes using different base models: VGG16, VGG19, ResNet50, InceptionV3, DenseNet121, and MobileNetV2. The project focuses on customizing these models and evaluating their performance in the multilabel classification task. 

## Dataset Source

The dataset used in this project can be found [here](https://www.kaggle.com/datasets/shrutisaxena/yoga-pose-image-classification-dataset).

Here's an overview of the steps involved in the image processing and model tuning process:

## Image Processing Steps
1. **Array NumPy Conversion**: The images were processed into NumPy arrays for efficient handling in deep learning models.
2. **Reshaping to 224x224**: The images were reshaped to a standardized dimension of 224x224 pixels.
3. **Conversion to RGB**: Greyscale photos were converted to RGB format to ensure consistency in input channels.
4. **Normalization**: The reshaped photos were normalized, scaling pixel values from 0 to 1.
5. **Label Encoding and One-Hot Encoding**: Labels were encoded using one-hot encoding to prepare them for multilabel classification.
6. **Train-Test Split**: The dataset was split into training and testing sets with an 80:20 ratio.

## Overview

In this project, we explore multilabel classification using state-of-the-art deep learning models. The goal is to accurately classify images into 107 distinct classes. Six base models were utilized for this task: VGG16, VGG19, ResNet50, InceptionV3, DenseNet121, and MobileNetV2. These models were customized to suit the multilabel classification problem, and their performance was evaluated in terms of accuracy.

## Base Models

- **VGG16**: A widely used convolutional neural network architecture.
- **VGG19**: An extended version of VGG16 with deeper layers.
- **ResNet50**: A residual network with 50 layers, enabling the training of very deep networks.
- **InceptionV3**: A model designed to improve efficiency and accuracy through specially crafted modules.
- **DenseNet121**: A densely connected convolutional network that connects each layer to every other layer in a feed-forward fashion.
- **MobileNetV2**: A lightweight model optimized for mobile and edge devices.

## Customization and Evaluation

Each base model was customized for the multilabel classification task. Customization involved adding specific layers to adapt the models to the problem's complexity. The models were then trained and evaluated using the provided dataset. The accuracy of each model was calculated to measure its performance.

## Results

The following table displays the accuracy achieved by each customized model:

| Model      | Accuracy |
|------------|----------|
| VGG16      | 91.74%   |
| VGG19      | 91.03%   |
| ResNet50   | 67.67%   |
| InceptionV3| 88.65%   |
| DenseNet121| 89.65%   |
| MobileNetV2| 89.70%   |

These results provide insights into the effectiveness of each model for the multilabel classification task. The accuracy values reflect the models' ability to correctly classify images into the 107 predefined classes.

## Model Customization and Hyperparameter Tuning
The project involved extensive hyperparameter tuning to optimize model performance. The ResNet-50 model with pre-trained weights was used as the base model. The following hyperparameters were explored:

### Hyperparameters Explored:
- **Learning Rates**: [0.0001, 0.001, 0.01, 0.0003, 0.003, 0.03]
- **Loss Functions**: ['categorical_crossentropy', 'mean_squared_error']
- **Optimizers**: [Adam, SGD, RMSprop]
- **Dropout Rates**: [0, 0.1, 0.3, 0.5]
- **Batch Sizes**: [8, 16, 32, 64, 128]

### Tuning Process:
1. **Freezing Pre-trained Layers**: Layers in the pre-trained model were frozen to retain pre-learned features.
2. **Hyperparameter Iteration**: The project systematically explored combinations of learning rates, loss functions, optimizers, dropout rates, and batch sizes.
3. **Custom Model Creation**: Custom models were created by adding dense layers, dropout, and batch normalization on top of the pre-trained ResNet-50.
4. **Model Compilation**: Each custom model was compiled with the current set of hyperparameters.
5. **Model Naming**: Models were named based on the tuned hyperparameters to identify each configuration.
6. **Model Storage**: Custom models along with their parameters were stored for further evaluation.

### Model Evaluation:
The project evaluated the performance of each tuned model in terms of accuracy on the test dataset. Models were systematically compared to identify the best configuration.

## Conclusion
This project demonstrates a systematic approach to multilabel classification using various base models and thorough hyperparameter tuning. The combination of standardized image preprocessing and extensive model tuning resulted in accurate multilabel classification. The results and configurations are available in this repository. Feel free to explore the code and experiment with additional configurations to further enhance the models' performance. If you have any questions or feedback, please don't hesitate to reach out. Thank you for your interest in this project!
