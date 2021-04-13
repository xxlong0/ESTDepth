# Multi-view Depth Estimation using Epipolar Spatio-Temporal Networks (ESTDepth)

### [Project Page](https://www.xxlong.site/ESTDepth/) | [Video]() | [Paper](https://arxiv.org/pdf/2011.13118) | [Data](#dataset)

<img src='docs/images/teaser.png'/>

We present a novel method for multi-view depth estimation from a single video, which is a critical task in various applications, such as perception, reconstruction and robot navigation. 
Although previous learning-based methods have demonstrated compelling results, most works estimate depth maps of individual video frames independently, without taking into consideration the strong geometric and temporal coherence among the frames. 
Moreover, current state-of-the-art (SOTA) models mostly adopt a fully 3D convolution network for cost regularization and therefore require high computational cost, thus limiting their deployment in real-world applications. 
Our method achieves temporally coherent depth estimation results by using a novel Epipolar Spatio-Temporal (EST) transformer to explicitly associate geometric and temporal correlation with multiple estimated depth maps. 
Furthermore, to reduce the computational cost, inspired by recent Mixture-of-Experts models, we design a compact hybrid network consisting of a 2D context-aware network and a 3D matching network which learn 2D context information and 3D disparity cues separately. 

Here is the official repo for the paper:

* [Multi-view Depth Estimation using Epipolar Spatio-Temporal Networks (Long et al., 2021, <span style="color:red">CVPR 2021</span>)](https://arxiv.org/pdf/2011.13118).


## Table of contents
-----
  * [Installation](#requirements-and-installation)
  * [Dataset](#dataset)
  * [Usage](#train-a-new-model)
    + [Training](#train-a-new-model)
    + [Evaluation](#evaluation)
  * [License](#license)
  * [Citation](#citation)
------