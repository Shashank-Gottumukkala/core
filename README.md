# core

This repository serves as a centralized hub for various utilities, pre-built models, diverse datasets, data augmentations, and other essential files required to facilitate seamless training and testing of deep learning models. This document provides an overview of the repository's contents and offers guidance on how to effectively leverage these resources.

## [Datasets](https://github.com/shashankg69/core/tree/main/Dataset)
The datasets module simplifies data loading, processing, and visualization. It includes a generic MyDataSet class capable of creating train and test data loaders. This class can easily handle transformations such as normalization and tensor conversion. While the module currently supports MNIST and CIFAR-10 datasets, extending it with additional datasets is straightforward.

## [models](https://github.com/shashankg69/core/tree/main/models)
The models module houses model definitions for various deep learning architectures. These models can serve as foundations for your projects or be customized to meet specific requirements.

## [utils](https://github.com/shashankg69/core/tree/main/utils)
The utils module provides a set of tools to facilitate the experimentation process, making training and testing efficient. It includes submodules for managing backpropagation, experiments, and miscellaneous functions.
