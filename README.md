# Cats-vs-Dogs-Kaggle-Project
Cats vs. Dogs Image Classification using a CNN
This repository contains the code and resources for a deep learning project focused on binary image classification. The goal is to accurately distinguish between images of cats and dogs using a Convolutional Neural Network (CNN) built with TensorFlow. This project was developed by Mandavi Singh as part of a machine learning curriculum.



Table of Contents
Project Overview

Dataset

Methodology

Data Preprocessing

Model Architecture

Training and Callbacks

Results and Performance

How to Use

Future Work

Contact

Project Overview
This project tackles the classic "Cats vs. Dogs" classification problem. A custom CNN was constructed and trained on a large dataset to learn the distinguishing features of each animal. The entire workflow was implemented within a Kaggle Notebook, leveraging a Tesla T4 GPU for efficient computation and training.


Technology Stack: Python, TensorFlow (Keras), Matplotlib, NumPy


Platform: Kaggle Notebooks 


Hardware: Tesla T4 GPU 

Dataset
The project utilizes the "Cats vs Dogs" dataset from a Microsoft-sponsored Kaggle competition.


Total Images: Approximately 25,000 JPG files.


Classes: Cat, Dog.


Class Distribution: The dataset is perfectly balanced, with 50% cat images and 50% dog images. This balance is critical for preventing model bias and ensuring reliable training and evaluation.



Data Split: The dataset was divided into three subsets:



Training Set: 80% (20,000 images).




Validation Set: 10% (2,500 images).




Test Set: 10% (2,500 images).


Methodology
Data Preprocessing
Before training, the data underwent several preprocessing steps:

Corrupted image files were identified and removed from the dataset.

All images were resized to a uniform dimension of 150x150 pixels.

The data was organized into separate subdirectories for the training, validation, and test sets to facilitate loading.

Images were loaded efficiently using TensorFlow's 

image_dataset_from_directory utility.

Model Architecture
A Sequential CNN model was designed to progressively extract hierarchical features from the images.


Structure: The architecture consists of multiple pairs of Conv2D layers followed by MaxPooling2D layers. A 

GlobalAveragePooling2D layer is used to reduce feature maps into a vector before passing them to the final dense layers.


Layers: The detailed layer configuration is as follows:

conv2d (32 filters)

conv2d_1 (64 filters)

max_pooling2d

conv2d_2 (64 filters)

conv2d_3 (128 filters)

max_pooling2d_1

conv2d_4 (128 filters)

conv2d_5 (256 filters)

global_average_pooling2d


dense (1024 neurons) 


dense_1 (1 neuron, sigmoid activation for binary output) 


Total Parameters: 837,121 (all are trainable).

Training and Callbacks

Optimizer: Adam.


Loss Function: Binary Crossentropy.


Batch Size: 32.


Epochs: The model was set to train for 50 epochs, but EarlyStopping was used to prevent overfitting.

Callbacks:


EarlyStopping: This callback monitored the validation loss (val_loss). Training would halt if no improvement was seen for 3 consecutive epochs (

patience=3) , and the best model weights were restored upon completion.


Results and Performance
The model achieved high accuracy and demonstrated good generalization on unseen data. The final trained model is saved as 

cats_dogs_classifier.h5.


Training and Validation Curves: The training and validation accuracy curves stayed close together, with validation accuracy peaking around 87%, indicating the model did not overfit. Both training and validation loss steadily decreased from ~0.7 to below 0.2.



Overall Test Accuracy: 84%.

Classification Report on Test Data:
| Class | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| Cat | 0.86  | 0.81  | 0.83  |




| Dog | 0.82  | 0.87  | 0.84  |



How to Use
Clone the repository:

Bash

git clone https://github.com/mandavi-singh/Cats-vs-Dogs-Classifier.git
cd Cats-vs-Dogs-Classifier
Install dependencies:

Bash

pip install tensorflow matplotlib numpy
Download the Dataset: Download the "Cats vs Dogs" dataset from Kaggle and place it in the appropriate directory structure as referenced in the notebook.

Run the Notebook: Open and run the cats_vs_dogs_classifier.ipynb notebook in a Jupyter or Kaggle environment.

Use the Saved Model: Load the cats_dogs_classifier.h5 model to make predictions on new images.

Future Work
Potential areas for future improvement include:


Data Augmentation: Applying techniques like rotation, flipping, and zooming to artificially expand the dataset and improve model robustness.


Transfer Learning: Utilizing powerful pre-trained models (like VGG16, ResNet, or EfficientNet) to leverage learned features and potentially boost accuracy.


Deployment: Building an interactive web application with Streamlit or Flask to deploy the model for real-world use.

Contact
Feel free to connect with me for any questions or collaborations.


Author: Mandavi Singh 


LinkedIn: www.linkedin.com/in/mandaviofficial 
Email: singhmandavi002@gmail.com 









