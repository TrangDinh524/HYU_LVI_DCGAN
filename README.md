# DCGAN_ImageGeneration_CIFAR100

This repository contains the implementation of training a DCGAN (Deep Convolutional Generative Adversarial Network) for image generation using the CIFAR-100 dataset. The model generates synthetic images by learning the distribution of real images from the CIFAR-100 classes.

## Requirements

To run this project, you need the following libraries:

- `torch`
- `torchvision`
- `matplotlib`
- `numpy`
- `scipy`
- `IPython`
- `tqdm`

Install the necessary libraries using:
pip install torch torchvision matplotlib numpy scipy ipython tqdm
## Hyperparameters

The following hyperparameters are used for training:

- Number of Epochs: 100
- Batch Size: 64
- Learning Rate: 0.0002
- Beta1: 0.5

## Model Architecture

The DCGAN consists of two main components:

**Generator:**
- **Transposed Convolutional Layers:** Upsample random noise into images, progressively increasing the image resolution.
- **Batch Normalization:** Stabilizes training by normalizing intermediate activations.
- **ReLU Activation:** Introduces non-linearity for better learning.
- **Tanh Activation (final layer):** Ensures the output images have pixel values in the range [-1, 1].

**Discriminator:**
- **Convolutional Layers:** Classifies input images as real or fake, progressively extracting features from the images.
- **Leaky ReLU Activation:** Helps with gradient flow during backpropagation.
- **Sigmoid Activation (final layer):** Outputs a probability score (real or fake).

## Train Model

To train the model, follow these steps:

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Configure the following hyperparameters and set relevant paths in the script:

    - Number of Epochs: 200  
    - Batch Size: 64
    - Learning Rate: 0.0002  
    - Beta1: 0.5  

  All hyperparameters are located at the top of the script for easy modification.
  
3. Train the model using:
    nohup python3 dcgan_3.py &

    If you want to check the process of the model during training, run 'tail nohup.out'
   
5. Model is automatically logged during training to folder:
- `nohup.out'
  
## Evaluation

After training, the model is evaluated on the test set. The following metrics are computed for evaluation:

- Image Quality (Visual inspection)
- Inception Score (IS)
- Frechet Inception Distance (FID)

## Random Seed

The code uses a fixed random seed for dataset splitting and transformations to ensure reproducibility:

- Random Seed: 999

To change the random seed, modify the variable in the script.

## Additional Information

- **Training Time:** Depending on your hardware, training may take several hours or days. It is recommended to use a machine with GPU support for faster training.
- **Dataset:** CIFAR-100 contains 60,000 images across 100 classes. The dataset can be loaded from `torchvision.datasets.CIFAR100`.
