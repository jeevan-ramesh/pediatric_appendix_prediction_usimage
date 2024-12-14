# pediatric_appendix_prediction_usimage
by Faseena Farsana K and Jeevan Ramesh


image detection part

# Pediatric Appendicitis Analysis Using Deep Learning

## Introduction
Appendicitis is a common medical condition that requires accurate diagnosis to prevent complications. This project aims to develop and compare different deep-learning models for the diagnosis of appendicitis using medical images. The models used in this study include:
- A custom Convolutional Neural Network (CNN)
- A VGG16-based model
- A ResNet50-based model

## Dataset
The dataset consists of medical images related to pediatric appendicitis cases. Each image is labeled to indicate whether the patient has appendicitis or not.

## Model Architectures

### 1. Custom CNN Model
**Architecture:**
- Resizing and Rescaling layer
- Conv2D layers: 32, 64, 128 filters with 3x3 kernels, ReLU activation
- MaxPooling2D layers after each Conv2D layer
- Flatten layer
- Dense layer with 128 units and ReLU activation
- Dropout layer with a 0.5 dropout rate
- Final Dense layer with 2 units and softmax activation

**Compilation:**
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Metrics: Accuracy

### 2. VGG16-Based Model
**Architecture:**
- Resizing and Rescaling layer
- Pre-trained VGG16 model without the top classification layer
- Flatten layer
- Dense layer with 128 units and ReLU activation
- Dropout layer with a 0.5 dropout rate
- Final Dense layer with 2 units and softmax activation

**Compilation:**
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Metrics: Accuracy

### 3. ResNet50-Based Model
**Architecture:**
- Resizing and Rescaling layer
- Pre-trained ResNet50 model without the top classification layer
- Flatten layer
- Dense layer with 128 units and ReLU activation
- Dropout layer with a 0.5 dropout rate
- Final Dense layer with 2 units and softmax activation

**Compilation:**
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Metrics: Accuracy

## Model Training and Evaluation
All models were trained on the same dataset with the same number of epochs and batch size. The performance of each model was evaluated using loss and accuracy metrics.

## Results
The table below summarizes the performance of each model:

| Model            | Loss   | Accuracy |
|------------------|--------|----------|
| Custom CNN       | 1.0517 | 58.24%   |
| VGG16-Based      | 0.4519 | 79.67%   |
| ResNet50-Based   | 0.6931 | 50.55%   |

## Discussion

### Custom CNN Model
The custom CNN model achieved a moderate accuracy of 58.24% but had a high loss of 1.0517, indicating room for improvement in its design and optimization.

### VGG16-Based Model
The VGG16-based model outperformed the other models with an accuracy of 79.67% and a lower loss of 0.4519. This suggests that the VGG16 architecture is well-suited for appendicitis diagnosis.

### ResNet50-Based Model
The ResNet50-based model did not perform as well, with an accuracy of 50.55% and a loss of 0.6931. This could be due to the complexity of the model or insufficient training data.

## Conclusion
Based on the results, the VGG16-based model is the most effective for diagnosing pediatric appendicitis in this study. Future work could focus on:
- Further fine-tuning the VGG16 model
- Exploring other pre-trained models to enhance accuracy
- Increasing the size of the dataset
- Applying advanced data augmentation techniques

---

### Authors and Acknowledgments
This project was developed as part of a research study on pediatric appendicitis diagnosis using deep learning. Special thanks to the medical professionals who provided the dataset and guidance.

### License
This project is open-source and available for use under the [MIT License](LICENSE).


