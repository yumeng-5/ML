# Handwritten Greek Letter Recognition using CNN

## Overview:
This script implements a CNN model using PyTorch to classify images of handwritten Greek letters.

## Features:
* **Data Handling:**
    * Loads data from CSV files(flatten images).
    * Splits data into training and testing sets.
    * Augments the test set with random noise samples.
* **CNN Model:** A custom PyTorch CNN model with convolutional layers, batch normalization, max pooling, dropout, and fully connected layers.
* **Image Preprocessing:**
    * Normalization (pixel values scaled to [0, 1]).
    * Feature Enhancement: Applies morphological erosion followed by Sobel edge detection to emphasize character strokes and shapes.
* **Training:**
    * Trains the CNN using Adam optimizer and CrossEntropyLoss.
    * Saves the trained model weights (`.pth` file).
    * Plots training loss and accuracy curves.
* **Evaluation:**
    * `test` function: Standard evaluation reporting accuracy, classification report, and confusion matrix.
    * `test_hard` function: Evaluates using a confidence threshold to flag low-confidence predictions as "unknown" class(-1). Reports metrics accordingly and plots a confusion matrix, plus a confidence score histogram.
* **Configure:**
    * Open the main Python script. Modify the control flags near the bottom.
    * `RETRAIN_MODEL = True` : Set to `True` to train the model from scratch. Set to `False` to skip training and load a previously saved model (`greek_letter_model.pth`).
    * `RUN_STANDARD_TEST = True`: Set to `True` to run the standard evaluation (`test` function).
    * `RUN_HARD_TEST = True`: Set to `True` to run the evaluation with unknown detection (`test_hard` function).
    * You may also adjust parameters like `epochs`, `batch_size`, `lr`, `confidence_threshold`, file paths, etc., within the script or function calls.

## Packages:
1. Pytorch
2. numpy
3. matplotlib
4. sklearn
5. skimage
