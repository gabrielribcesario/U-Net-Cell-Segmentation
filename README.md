# Cell Tracking Challenge submission

My submission for the [Cell Tracking Challenge](https://celltrackingchallenge.net/) under the segmentation category for the [DIC-C2DH-HeLa dataset](https://celltrackingchallenge.net/2d-datasets/).

# Architecture and training

I'm pretty much using the same network architecture as described in the [original U-Net paper](https://arxiv.org/abs/1505.04597) (Ronneberger et al., 2015), except for the zero padding after each convolutional layer. The binary cross-entropy loss with the distance-based pixel weight maps described in the paper has also been implemented. I've also experimented with different data augmentation methods, mainly geometric variations such as rotations and [elastic deformations](https://ieeexplore.ieee.org/document/1227801) (Simard et al., 2003). To remedy problems with gray value variations I've added a [contrast limited adaptive histogram equalization (CLAHE)](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html) pre-processing step.

