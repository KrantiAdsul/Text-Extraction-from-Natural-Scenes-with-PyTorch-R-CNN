# Text-Extraction-from-Natural-Scenes-with-PyTorch-R-CNN

## Overview
This project focuses on fine-tuning a PyTorch R-CNN (Region-based Convolutional Neural Network) model to extract text regions from natural scenes. The objective is to create a model that can accurately detect and localize text in images. The project includes the following key components:

## Dataset
- Utilize the Text MS-COCO dataset, a subset of the MS-COCO dataset with additional text annotations. This dataset provides images with text regions and corresponding bounding box annotations.
- Download the dataset from [here](https://data.usc-ece.com) and extract the images.
- Access the JSON annotation file containing target bounding boxes and text content.

## Preparation
- Prepare validation figures that compare machine-predicted bounding boxes with ground-truth boxes.
- Generate loss curves to visualize component-wise R-CNN losses as a function of training epochs.

## Model Selection and Adaptation
- Choose a suitable pre-trained torchvision model(here: 'fasterrcnn_resnet50_fpn' used which is a Faster R-CNN model based on the ResNet-50 architecture pretrained on the COCO (Common Objects in Context) dataset here adapted to detect 2 classes - objects and background) while considering training resources.

## Setup
- Define data transformations, such as normalization, as needed.
- Create a dataset, e.g., CocoDataset, with the specified transformations.
- Set up a data loader, e.g., DataLoader, with a collate function if required.
- Instantiate the R-CNN model using the get_model() function.

## Training
- Specify the optimizer and, if applicable, a learning rate scheduler.
- Train the model over a specified number of epochs.
- Implement the training loop, including inference, backpropagation, and model parameter updates.
- Monitor and record the four component-wise RCNN losses (e.g., classification, localization) as a function of epoch.
- Implement model checkpointing to save weights and optimizer states for resuming training if necessary.
- Plot each of the four RCNN loss components against the epoch number.

## Evaluation and Visualization
- Implement the evaluation function evaluate() to visualize bounding boxes and evaluate model performance.
- Use the evaluation function to validate the model's accuracy in localizing text regions.
- Utilize four-panel subplots to visualize machine-predicted bounding boxes alongside ground-truth boxes.
