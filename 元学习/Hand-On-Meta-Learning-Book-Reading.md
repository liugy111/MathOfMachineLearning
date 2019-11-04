[TOC]

# 1. Overview

Meta Learning 有三个方向：

- Learning the metric space
  - 学习不同任务/度量空间 Metirc Space 之间的相似性
  - 例如 Siamese networks, prototypical networks, and relation networks. 
- Learning the initializations
  - 学习如何不通过梯度下降，从而直接找到optimal weight的方法
  - MAML, Reptile, and Meta-SGD 
- Learning the optimizer
  - 在Few-shot学习中，GD是没法用的
  - In this case, we will learn the optimizer itself. **We will have two networks: a base network that actually tries to learn and a meta network that optimizes the base network. **

## 1.1 Learning to learn gradient descent by gradient descent

对应了 Learning the optimizer 的范畴，meta net 是一个RNN。

Our optimizee (base network) is optimized through our optimizer (RNN, optimized by gradient descent)。

Optimizer RNN的损失函数：
$$
L(\phi)=\mathbb{E}_f[f(\theta(f,\phi))]\\
$$

> Loss = Average Loss of the optimizee (base network)
>
> \phi for RNN params, \theta for base network f 's params

Optimizer RNN的参数更新（梯度下降法）：
$$
(g_t,h_{t+1}) =m(\nabla_t, h_t, \phi)
$$

- RNN的输入：
  - \nabla_t: 在t时刻，optimizee 执行support set 的某任务时，损失函数计算出来的梯度。也就是说，$\nabla_t = \nabla_{t} f(\theta_t)$

  - h_t: 当前RNN的hidden state

  - \phi: 当前RNN的参数

- RNN的输出：
  - g_t: 代替GD，给 optimizee 更新的梯度，从而使 $\theta_{t+1} = \theta_t + g_t$

  - h_{t+1}: next state of RNN

## 1.2 Optimization as a model for few-shot learning

对于Few-shot Learning，用LSTM可以替代GD，以LSTM的cell直接作为base network update，也不需要 learning rate 了。

对于LSTM的forget gate:

![](imgs/0.png)

对于LSTM的input gate:

![](imgs/1.png)

LSTM的cell:

![](imgs/2.png)

# 2. Face and Audio Recognition Using Siamese Networks （Python）

## 孪生网络

两个网络的结构和参数完全相同，对不同的输入会前向输出不同的embedding（feature vector），特征向量的相似性反映了输入的相似性。

相似性判断函数被称为 energy function，常见的有欧氏距离和余弦相似度。

孪生网络输入：The input to the siamese networks should be in pairs, **(X1, X2)**, along with their binary label, **Y ∈ (0, 1)**, stating whether the input pairs are a **genuine pair (same)** or an **imposite pair (different)**.  

孪生网络损失函数：
$$
\text{Contrasive Loss} = Y(E)^2+(1-Y)\max(margin-E, 0)^2
$$

- The term **margin **is used to hold the constraint, that is, when two input values are dissimilar, and if their distance is greater than a margin, then they do not incur a loss. 

# 3. Prototypical Networks and Their Variants

## 原型网络

常用在few-shot的分类任务中。

用CNN提取图片的特征，对同一个类别的特征向量取均值作为整个类的 class prototype。

对于query data，计算其特征与各个class prototype的欧氏距离，softmax to this distance and get the probabilities.  