# Decoding Neural Brain Activity for Image Classification

Fachpraktikum Machine learning and Computer vision laboratory for Human Computer Interaction

## Abstract

Methods such as function magnetic resonance imaging (fMRI) allow to create an image of the human brain showing neural activity. Previous work tried to reconstruct the stimuli shown to a participant from their fMRI data. The reconstructed images are often lacking in image quality or do not represent the correct image class. The result is easier to understand if instead of reconstructing the actual image we train models that predict the class of a stimuli from fMRI data. In this paper we propose two deep learning models: Long-Short Term Memory (LSTM) and 3D-Convolutional Neural Network (3D-CNN) to decode the fMRI data and predict the class label of the stimuli.

## Introduction

This repository contains code for the implementation of LSTM and CNN models for classification of visual stimuli from fMRI data on BOLD5000 dataset. 

## Requirements

Use the package manager [Anaconda](https://docs.anaconda.com/anaconda/install/) to install all the dependencies from requirements.txt

```bash
conda create --name <env> --file requirements.txt
```

## Dataset

Pre-processed and raw BOLD5000 dataset is stored in /bigpool/export/users/datasets_faprak2020/BOLD5000/ and its subfolders. The processed data used for models are stored under respective model names in the parent directory.

## Usage

### LSTM
Run the notebook: lstm/lstm_classifier.ipynb

### CNN
Run the script: cnn/main.py
```bash
python3 main.py --epochs N -b N --early_stopping N --num_workers NUM_WORKERS --optimizer {Adam,SGD} --lr LR --weight_decay WEIGHT_DECAY
```

## References

1. Dataset was obtained from BOLD5000 available [here](https://figshare.com/articles/dataset/BOLD5000/6459449/) 
2. Python Code for LSTM model was based on this [code](https://github.com/arashjamalian/fmriNet)

