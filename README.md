# Sports Ball Classification Using ResNet18

## Overview

This project is a multi-class image classification system that identifies the type of sports ball in an image. It was developed as part of a Computer Vision assignment for the Daneshkar AI Bootcamp, based on a sports equipment store use case.

The model classifies images into **15 different ball categories** using transfer learning with a pretrained ResNet18 model in PyTorch.

## Objectives

* Explore and preprocess the image dataset.
* Apply data augmentation to improve model generalization.
* Build and train a multi-class image classification model.
* Evaluate model performance using validation and test data.
* Analyze classification errors and identify challenging classes.

## Dataset

The dataset contains images organized into class-specific folders.

The 15 classes are:

1. American football
2. Baseball
3. Basketball
4. Billiard ball
5. Bowling ball
6. Cricket ball
7. Football
8. Golf ball
9. Hockey ball
10. Hockey puck
11. Rugby ball
12. Shuttlecock
13. Table tennis ball
14. Tennis ball
15. Volleyball

Dataset directory structure:

```text
Project_Dataset/
├── train/
│   ├── basketball/
│   ├── football/
│   └── ...
└── test/
    ├── basketball/
    ├── football/
    └── ...
```

Each class has its own folder, which is automatically interpreted as a label by PyTorch's `ImageFolder`.

## Technologies

* Python
* PyTorch
* Torchvision
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Jupyter Notebook

## Methodology

### 1. Data Exploration and Preprocessing

The dataset was explored to understand class distribution and image characteristics. Images were checked for potential quality issues, and preprocessing was applied to prepare them for model training.

### 2. Data Augmentation

Training images were augmented using transformations such as:

* Random resized cropping
* Random horizontal flipping
* Random rotation
* Color jitter

These transformations introduce variation into the training data and help the model generalize to unseen images.

Validation and test images use deterministic preprocessing without random augmentation.

### 3. Transfer Learning

Transfer learning was used because the dataset is relatively small compared with the large datasets commonly used to train deep neural networks from scratch.

The pretrained model has already learned useful visual features from ImageNet, such as edges, textures, and shapes. Reusing these features reduces training time and computational requirements.

### 4. Model Selection: ResNet18

ResNet18 was selected because it offers a balance between model complexity, computational efficiency, and classification capability.

Its residual connections help the network learn deeper feature representations. The pretrained ImageNet weights also make it suitable for transfer learning on this dataset.

### 5. Fine-Tuning

The pretrained ResNet18 model was adapted to the 15-class classification task.

* Most pretrained layers were frozen.
* The final convolutional block (`layer4`) was unfrozen for fine-tuning.
* The original fully connected classifier was replaced with a new classifier for 15 classes.
* Dropout was included to help reduce overfitting.

### 6. Training

The model was trained using:

* **Loss function:** Cross-Entropy Loss
* **Optimizer:** AdamW
* **Learning-rate scheduling:** ReduceLROnPlateau (where configured)
* **Regularization:** Weight decay and dropout
* **Early stopping:** To stop training when validation performance stops improving

The model checkpoint with the highest validation accuracy was saved and restored for evaluation.

## Results

The best validation accuracy obtained in the reported experiments was approximately **87.6%**.

| Metric                   |   Result |
| ------------------------ | -------: |
| Best validation accuracy |   87.64% |
| Number of classes        |       15 |
| Model                    | ResNet18 |

The validation result is based on the best checkpoint selected during training. Test performance should be reported separately after evaluating that checkpoint on the held-out test dataset.

## Error Analysis

The model may confuse visually similar categories, such as:

* Football, American football, and rugby ball
* Tennis ball and table tennis ball
* Hockey ball and hockey puck

A confusion matrix, classification report, and inspection of misclassified images can help identify which classes are most challenging and where further improvements may be useful.

## Project Structure

```text
sports-ball-classification/
├── cv_project.ipynb
├── Project_Dataset/
│   ├── train/
│   └── test/
└── README.md
```

*The dataset is not necessarily included in this repository.*

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/sanamasghari/sports-ball-classification.git
   ```

2. Navigate to the project directory:

   ```bash
   cd sports-ball-classification
   ```

3. Install the required libraries:

   ```bash
   pip install torch torchvision numpy pandas matplotlib scikit-learn jupyter
   ```

4. Make sure the dataset is available in the expected `Project_Dataset/` directory.

5. Open and run the notebook:

   ```bash
   jupyter notebook cv_project.ipynb
   ```

## Future Improvements

* Perform detailed per-class error analysis.
* Experiment with different augmentation strategies.
* Compare ResNet18 with other pretrained architectures.
* Tune hyperparameters and fine-tuning strategies.
* Evaluate the final model on the held-out test dataset.

## Author

**Sanam Asghary**

Junior AI Engineer with hands-on experience in Python and AI projects.

[GitHub](https://github.com/sanamasghari/resnet18-ball-classification)

[LinkedIn](https://www.linkedin.com/in/sanamasghari/)
