[TOC]

# activation functions

![](imgs/1.png)

### sigmoid 的问题 (Don’t use sigmoid in practice)

1. Saturated neurons “kill” the gradients：把output压缩到0-1区间对于x=-100/100这种会损失梯度
2. Sigmoid outputs are not zero-centered：output always positive --> gradients always all positive or negative (For a single element! Minibatches help)
3. exp() is a bit compute expensive

### ReLU 的优点

1. Does not saturate (in +region)
2. Very computationally efficient
3. Converges much faster than sigmoid/tanh in practice (e.g. 6x)

### ELU: 似乎更靠谱

1. All benefits of ReLU
2. Closer to zero mean outputs
3. Negative saturation regime compared with Leaky ReLU adds some robustness to noise 

# data preprocessing

![](imgs/2.png)

> `axis=0` since each example in a row

### Data Normalization

![](imgs/3.png)

### In practice for Images: center only

![](imgs/4.png)

# weight initialization

### Interesting Works

1. Fixup Initialization: Residual Learning Without Normalization, Zhang et al, 2019
2. The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks, Frankle and Carbin, 2019

# Batch Normalization

![](imgs/5.png)

### BN in test-time

![](imgs/6.png)

### BN in practice

![](imgs/7.png)

Usually inserted after Fully Connected or Convolutional layers, and before nonlinearity.

### Layer Normalization

![](imgs/8.png)

### Instance Normalization

![](imgs/9.png)

### Group Normalization

![](imgs/10.png)

# Babysitting the Learning Process

# Hyperparameter Optimization