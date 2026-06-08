# Gemstone-Classification-Using-Deep-Learning
Automated gemstone classification using deep learning. This project compares CNN, Xception, and ResNet50V2 models on an 87-class dataset, achieving 65.56% accuracy with Xception and highlighting the benefits of transfer learning.
# Gemstone Classification Using Deep Learning

## Overview

Gemstones play a significant role in the jewelry industry, where accurate identification is essential for determining their value, rarity, and authenticity. However, many gemstones exhibit similar visual characteristics, making manual classification challenging and prone to errors.

This project presents a comparative study of deep learning models for automated gemstone classification. A dataset containing 87 gemstone categories was used to train and evaluate three models: a custom Convolutional Neural Network (CNN), Xception, and ResNet50V2. Transfer learning techniques were employed to improve classification performance and reduce training time.

The experimental results demonstrate that pretrained deep learning architectures outperform a traditional CNN. Among the evaluated models, Xception achieved the highest classification accuracy of 65.56%, followed closely by ResNet50V2 with 65.29%, while the custom CNN achieved 54.55%.

## Key Features

* Classification of 87 gemstone categories
* Comparative analysis of CNN, Xception, and ResNet50V2
* Transfer learning with ImageNet pretrained weights
* Fine-tuning for performance optimization
* Accuracy and loss visualization
* Research-oriented implementation using TensorFlow and Keras

## Results

| Model      | Accuracy |
| ---------- | -------- |
| CNN        | 54.55%   |
| Xception   | 65.56%   |
| ResNet50V2 | 65.29%   |

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Google Colab

## Future Work

Future improvements may include larger gemstone datasets, advanced data augmentation techniques, ensemble learning approaches, and real-time gemstone recognition systems.
