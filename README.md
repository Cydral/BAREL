# BAREL
<i>Backpropagation And Reinforcement-based Environment for Learning</i> 
[![Publication](https://img.shields.io/badge/Publication-PDF-blue.svg)](https://github.com/Cydral/BAREL/blob/main/publication/CTY-I2A-20230403.pdf)

BAREL is a novel hybrid learning approach that combines self-supervised learning with supervised reinforcement training for image content understanding tasks. The method aims to leverage the strengths of both self-supervised and supervised learning while minimizing data labeling efforts.

## Overview

The BAREL approach unfolds in two main learning phases:

1. **Self-Supervised Learning**: in this phase, a convolutional neural network (CNN) backbone is trained using a self-supervised learning method, such as Barlow Twins, on a large dataset of unlabeled images. This stage aims to establish primary knowledge and pre-structure the weight distribution of the network.

2. **Supervised Reinforcement Training**: after the self-supervised learning phase, the same CNN backbone is fine-tuned using a smaller set of labeled data and a supervised metric learning method, such as Triplet Loss. This phase is designed to reinforce and specialize the acquired knowledge for a specific task, like facial recognition.

The BAREL approach demonstrates promising results, outperforming both supervised and self-supervised learning methods alone, while requiring fewer labeled data. It suggests a more realistic and human-like learning approach for visual content understanding.

## Key Features

- **Two-Phase Learning**: Combines self-supervised learning and supervised reinforcement training to leverage the strengths of both approaches.
- **Knowledge Transfer**: The same CNN backbone is used throughout the learning process, enabling knowledge transfer and consolidation.
- **Reduced Data Labeling**: Requires a smaller set of labeled data compared to traditional supervised learning methods.
- **Improved Generalization**: The self-supervised learning phase helps establish primary knowledge and improve generalization.
- **Synthetic Data Usage**: Utilizes synthetic data generated by GAN for the self-supervised learning phase, addressing data privacy concerns.
- **Open Source**: The models are distributed as open-source, allowing the community to refine them for specific use cases and contribute to addressing algorithmic biases.

## Getting Started

The BAREL approach and its implementation details are thoroughly described in the attached publication in "publication/CTY-I2A-20230403.pdf". To get started with this method, you can therefore access the publication and explore the provided details.

This project's GitHub repository also includes the training codes for the three presented models (supervised, self-supervised, and BAREL) and compiled binaries for Windows 64-bit systems (CUDA required). This will enable reproducing the results or utilizing the same principles to develop new models. Additionally, pre-trained models are available for download.

## License

The BAREL method and resources such as trained models are licensed under the [MIT License](https://github.com/Cydral/BAREL/blob/main/LICENSE).

## Acknowledgments

We would like to express our gratitude to the researchers and developers who contributed to the development of the self-supervised learning methods, such as Barlow Twins, and the convolutional neural network architectures, like ResNet and DenseNet, which serve as the foundation for the BAREL approach.

Additionally, we appreciate the efforts of the open-source community in providing valuable resources and tools, such as the Dlib library, which facilitated the implementation and evaluation of the BAREL method.
