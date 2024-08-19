

# SpecMet: A Deep Learning Method for Estimating Soil Heavy Metal Concentrations Using Hyperspectral Data

## Introduction

Soil heavy metal contamination has emerged as a global environmental concern, posing significant risks to human health and ecosystem integrity. Hyperspectral technology, with its non-invasive, non-destructive, large-scale, and high spectral resolution capabilities, shows promising applications in monitoring soil heavy metal pollution.

This repository contains the implementation of SpecMet, a novel deep learning method for estimating heavy metal concentrations in naturally contaminated agricultural soils using hyperspectral data. The SpecMet model integrates convolutional neural networks (CNNs), attention mechanisms, and graph neural networks to achieve end-to-end prediction of soil heavy metal concentrations.

## Directory Structure

- `data/`: Contains the hyperspectral data used for training and testing the model.
- `models/`: Pre-trained models and model checkpoints.
- `test/`: Code and scripts for testing the model on new data.
- `train/`: Scripts for training the SpecMet model.
- `utils/`: Utility scripts for data preprocessing and other helper functions.
- `README.md`: This file, providing an overview of the project.
- `main.py`: The main script to run the training and evaluation of the model.
- `西咸光谱.rar`: Compressed file containing additional spectral data for the study.

## Prerequisites

Before running the code, ensure you have the following dependencies installed:

- Python 3.x
- PyTorch
- NumPy
- Scikit-learn
- Matplotlib
- Any other dependencies listed in `requirements.txt`

You can install the required packages using:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the SpecMet model on the provided hyperspectral data:

```bash
python main.py --mode train --data_dir ./data --output_dir ./models
```

### Testing the Model

To test the trained model on new data:

```bash
python main.py --mode test --data_dir ./data --model_path ./models/model_checkpoint.pth
```

### Customizing the Model

You can modify the model architecture, training parameters, or other configurations in the `main.py` script according to your needs.

## Data

The hyperspectral data used in this study is provided in the `data/` directory. The additional spectral data is available in the `西咸光谱.rar` file. Extract the contents before use.

## Results

The SpecMet model has demonstrated superior performance in predicting heavy metal concentrations in soil, significantly outperforming traditional machine learning methods.

## Contact

For any questions or issues, please contact the project maintainers.

