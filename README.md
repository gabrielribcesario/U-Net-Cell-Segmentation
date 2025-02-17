# Cell Tracking Challenge submission

My submission for the [Cell Tracking Challenge](https://celltrackingchallenge.net/) under the segmentation category for the [DIC-C2DH-HeLa dataset](https://celltrackingchallenge.net/2d-datasets/).

# Architecture and training

I'm pretty much using the same network architecture as described in the [original U-Net paper](https://arxiv.org/abs/1505.04597) (Ronneberger et al., 2015), except for the zero padding after each convolutional layer. The binary cross-entropy loss with the distance-based pixel weight maps described in the paper has also been implemented. I also experimented with different data augmentation methods, mainly geometric variations such as rotations and [elastic deformations](https://ieeexplore.ieee.org/document/1227801) (Simard et al., 2003). I ended up adding a [contrast limited adaptive histogram equalization (CLAHE)](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html) pre-processing step to remedy problems with gray value variations, reducing the need of yet another form of data augmentation.

# Results

The original training dataset is composed of two different recordings. I used each of the recordings as a training-validation pair in a round-robin fashion, basically GroupKFold cross-validation. The results were:

- Average out-of-fold IoU: 0.844340
- Average out-of-fold dice loss: 0.141248

I'm yet to submit my results to the Cell Tracking Challenge organizers.