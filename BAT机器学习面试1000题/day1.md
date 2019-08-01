[TOC]

# 简要介绍SVM 

> Reference: wikipedia

SVM是一个面向数据的分类算法，它的目标是为确定一个分类超平面，从而将不同的数据分隔开。

SVM模型是将实例表示为空间中的点，这样映射就使得单独类别的实例被尽可能宽的明显的间隔分开。然后，将新的实例映射到同一空间，并基于它们落在间隔的哪一侧来预测所属类别。

除了进行线性分类之外，SVM 还可以使用所谓的核技巧有效地进行非线性分类，将其输入隐式映射到高维特征空间中。

任何超平面都可以写作满足  $\vec{w}\cdot\vec{x}-b=0$ 的点集。其中 ![{\displaystyle {\vec {w}}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/8b6c48cdaecf8d81481ea21b1d0c046bf34b68ec)（不必是归一化的）是该法向量。参数 ![{\displaystyle {\tfrac {b}{\|{\vec {w}}\|}}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/b4203b6e269e720d13207a93a931418fc6dac9f0) 决定从原点沿法向量 ![{\displaystyle {\vec {w}}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/8b6c48cdaecf8d81481ea21b1d0c046bf34b68ec) 到超平面的偏移量。

优化目标为：在 ![{\displaystyle y_{i}({\vec {w}}\cdot {\vec {x_{i}}}-b)\geq 1}](https://wikimedia.org/api/rest_v1/media/math/render/svg/f660f0bfa3334f273065d844154fff3122204f78) 条件下，最小化![{\displaystyle \|{\vec {w}}\|}](https://wikimedia.org/api/rest_v1/media/math/render/svg/5b6f27a892f3053ef0bfe273f88f18351a1a18ac)，对于![{\displaystyle i=1,\,\ldots ,\,n}](https://wikimedia.org/api/rest_v1/media/math/render/svg/520ffef648f7b26db5bae564be860346630635fc)

为了将SVM扩展到数据线性不可分的情况，我们引入Hinge Loss：

![{\displaystyle \max \left(0,1-y_{i}({\vec {w}}\cdot {\vec {x_{i}}}-b)\right).}](https://wikimedia.org/api/rest_v1/media/math/render/svg/8d0ce7194f9a86f19ddff06f3f423c9e770a1bac)

此时优化目标函数为：

![{\displaystyle \left[{\frac {1}{n}}\sum _{i=1}^{n}\max \left(0,1-y_{i}({\vec {w}}\cdot {\vec {x_{i}}}-b)\right)\right]+\lambda \lVert {\vec {w}}\rVert ^{2},}](https://wikimedia.org/api/rest_v1/media/math/render/svg/edfd21a4d3cb290e872f527487f0df3d29a90ce7)

# TensorFlow 计算图

Tensorflow 是基于图 (Graph) 的计算框架，图的节点由事先定义的运算 (操作、Operation) 构成，图的各个节点之间由张量 (tensor) 来链接，Tensorflow 的计算过程就是张量 (tensor) 在节点之间从前到后的流动传输过程。

有向图中，节点通常代表数学运算，边表示节点之间的某种联系，它负责传输多维数据 (Tensors)。

节点可以被分配到多个计算设备上，可以异步和并行地执行操作。因为是有向图，所以只有等到之前的入度节点们的计算状态完成后，其后的节点才能执行操作。推广到神经网络中，同一层之间的不同节点上的运算可以异步或并行的执行，但是前后层之间的执行还是要顺序执行，因为后一层的输入依赖于前一层的输出。