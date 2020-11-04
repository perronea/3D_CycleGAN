# Generative Adversarial Network for 3D Image-to-Image Translation

The purpose of this network is to train a model that creates T2w images from T1w images or vice versa. It takes as input full 3D NIfTIs and is capable of directly generating the 3D volume of the corresponding modality. 

This repository is based off of the CycleGAN arcitecture described by Per Welander and Anders Eklund in ["Generative Adversarial Networks for Image-to-Image Translation on Multi-Contrast MR Images - A Comparison of CycleGAN and UNIT"](https://arxiv.org/abs/1806.07777)

One of the biggest challenges in training this network is fitting it all in memory. A GPU is necessary to train a model in a reasonable amount of time (under a week), but this speed up comes at the cost of RAM. To train this 3D GAN we used a single NVIDIA Tesla V100 with 32GB of RAM. Anything less will likely result in Out Of Memory errors.


## System Setup

Training was done on a NVIDIA Tesla V100 with CUDA Version 10.2.89

Required python package versions are saved in requirements.txt and should be installed within a virtual environment
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt  
```
In addition to the requirements.txt, keras-contib needs to be installed from source. See [Keras-contrib Installation Instructions](https://github.com/keras-team/keras-contrib)

## Data Setup

The dataset should consist of paired T1w and T2w NIfTIs that have been aligned to eachother and skull stripped ideally using the same brain mask.

Move all T1w images to data/NAME_OF_DATASET/trainA and all T2w images to data/NAME_OF_DATASET/trainB

Paired data should be ordered the same in both directories. A data generator will read the paired data into memory one batch at a time so that datasets with hundreds or thousands of volumes can be used.


## Data Preparation

In order to minimize the input size during training we recommend first running prepare_train_data.py which crops each image to remove as much blank space as possible and then returns the smallest possible input image size for training. This processes copies a cropped version of all data to data/NAME_OF_DATASET_crop

After cropping, create data/NAME_OF_DATASET_crop/testA and data/NAME_OF_DATASET_crop/testB and move at least 10% of paired training data into the corresponding test directories.

## Input Size and Neural Network Architecture 

This deep learning model is pushing the limit on what can fit in memory. Through experimentation larger neural networks (more layers with more filters) have been shown to be more effective, however the network size is severely limited by the size of the input data. Additionally, given the number of convolutions, kernel size, and size of the strides between kernels only certain input shapes are allowed. Currently this requires a bit of guess and check to get it right. We recommend starting with the default generator and descriminator networks and the output crop from prepare_train_data.py. If you get an error that states a mismatch in one dimension then increase the size of that input dimension and try it again. If you get an Out Of Memory (OOM) error then decrease number of filters in a layer, or remove a layer entirely until it fits in your GPU's RAM.

## Generate synthetic images

Generate synthetic images by following specifications under:

* generate_images/ReadMe.md


### TODO

* Abstract Generator and Descriminator network creation so that optimal networks can be created for the given input shape
** OR make model independent of input shape
* Calculate acceptable input shapes given a network architecture 
* Calculate the size of memory required given a network, dataset, and hyperparameters
* 3D visualizations of training process
* Analysis on synthetic data




