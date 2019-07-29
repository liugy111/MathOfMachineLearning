# Caffe Directory Architecture

```
build
caffe.cloc
cmake
CMakeLists.txt
CONTRIBUTING.md
CONTRIBUTORS.md
data
    - cifar10
    - ilsvrc12
    - mnist
distribute
docker
docs
examples
    - 00-classification.ipynb
    - 01-learning-lenet.ipynb
    - 02-fine-tuning.ipynb
    - brewing-logreg.ipynb
    - cifar10
        - cifar10_full.prototxt
        - cifar10_full_sigmoid_solver_bn.prototxt
        - cifar10_full_sigmoid_solver.prototxt
        - cifar10_full_sigmoid_train_test_bn.prototxt
        - cifar10_full_sigmoid_train_test.prototxt
        - cifar10_full_solver_lr1.prototxt
        - cifar10_full_solver_lr2.prototxt
        - cifar10_full_solver.prototxt
        - cifar10_full_train_test.prototxt
        - cifar10_quick.prototxt
        - cifar10_quick_solver_lr1.prototxt
        - cifar10_quick_solver.prototxt
        - cifar10_quick_train_test.prototxt
        - convert_cifar_data.cpp
        - create_cifar10.sh
        - readme.md
        - train_full.sh
        - train_full_sigmoid_bn.sh
        - train_full_sigmoid.sh
        - train_quick.sh
    - CMakeLists.txt
    - cpp_classification
    - detection.ipynb
    - feature_extraction
    - finetune_flickr_style
    - finetune_pascal_detection
    - hdf5_classification
    - imagenet
    - images
    - mnist
    - net_surgery
    - net_surgery.ipynb
    - pascal-multilabel-with-datalayer.ipynb
    - pycaffe
    - siamese
    - web_demo
include
INSTALL.md
LICENSE
Makefile
Makefile.config
Makefile.config.example
matlab
models
python
README.md
scripts
src
tools
```

# Caffe的模块 

> Blobs, Layers, and Nets: anatomy of a Caffe model

### Blob

Blob 四维连续数组，通常表示为（n, k, w, h）

For example, in a 4D blob, the value at index (n, k, h, w) is physically located at index
$$
((n * K + k) * H + h) * W + w
$$
Blob 是基础的数据结构，可表示输入输出数据，也可表示参数数据。

Blob 对于图像分类问题，一般为 4D，但是对于全连接层 Blob 是 2D

a Blob stores two chunks of memories, *data* and *diff*. The former is the normal data that we pass along, and the latter is the gradient computed by the network.

> 翻译: 定义了前向 data 和反向 diff 两种存储空间

there are two different ways to access them: the const way, which does not change the values, and the mutable way, which changes the values.

> 翻译: 提供了 training 和 inference 两种访问方法

a Blob uses a SyncedMem class to synchronize values between the CPU and GPU in order to hide the synchronization details and to minimize data transfer.

> 翻译: 类 SyncedMem 封装了cpu/gpu数据同步的细节

### Layer

网络基本单元，每一层类型定义了3种计算: 初始化网络参数; 前向传播的实现; 后向传播。

![](https://caffe.berkeleyvision.org/tutorial/fig/layer.jpg)

| Layer Catalogue            | Examples                                                     |
| -------------------------- | ------------------------------------------------------------ |
| Data Layers                |                                                              |
| Vision Layers              |                                                              |
| Recurrent Layers           |                                                              |
| Common Layers              | Inner Product (fully connected layer); Dropout; Embed (for learning embeddings of one-hot encoded vector (takes index as input)) |
| Normalization Layers       |                                                              |
| Activation / Neuron Layers | 各种激活函数; Threshold Layer(performs step function at user defined threshold); Bias(adds a bias to a blob that can either be learned or fixed) |
| Utility Layers             | concatenate; mask; ArgMax; SoftMax                           |
| Loss Layers                |                                                              |

### Nets

Net 无回路有向图 (DAG)，有一个初始化函数，主要有两个作用： 1. 创建blobs和layers。2. 调用layers的setup函数来初始化layers。还有两个函数 Forward和Backward，分别调用layers的 forward 和 backward。

### Model Format

训练好的caffe model 是用于保存和恢复网络参数，后缀为 .caffemodel

solver保存和恢复运行状态，后缀为 .solverstate

The models are defined in plaintext protocol buffer schema (prototxt) while the learned models are serialized as binary protocol buffer (binaryproto) .caffemodel files.

### Data Processing

Caffe也不是直接处理原始数据的，而是由预处理程序将原始数据变换存储为LMDB或者LevelDB格式，这两种方式可保持较高的IO效率，加快训练时的数据加载速度，另一方面是因为数据类型很多（二进制文件，文本文件，JPG等图像文件等）不可能用同一套代码实现所有类型的输入数据读取，所以转换为统一的格式可以简化数据读取层的实现。

For MNIST:

```
layer {
  name: "mnist"
  # Data layer loads leveldb or lmdb storage DBs for high-throughput.
  type: "Data"
  # the 1st top is the data itself: the name is only convention
  top: "data"
  # the 2nd top is the ground truth: the name is only convention
  top: "label"
  # the Data layer configuration
  data_param {
    # path to the DB
    source: "examples/mnist/mnist_train_lmdb"
    # type of DB: LEVELDB or LMDB (LMDB supports concurrent reads)
    backend: LMDB
    # batch processing improves efficiency.
    batch_size: 64
  }
  # common data transformations
  transform_param {
    # feature scaling coefficient: this maps the [0, 255] MNIST data to [0, 1]
    scale: 0.00390625
  }
}
```

# 计算图

> [Forward and Backward](https://caffe.berkeleyvision.org/tutorial/forward_backward.html)

The `Net::Forward()` and `Net::Backward()` methods carry out the respective passes while `Layer::Forward()` and `Layer::Backward()` compute each step.

> 翻译: Net::For/Back() 与 Layer::For/Back() 是不同层面的函数，Net 类定义的前/后向更高层

Every layer type has `forward_{cpu,gpu}()` and `backward_{cpu,gpu}()` methods to compute its steps according to the mode of computation. A layer may only implement CPU or GPU mode due to constraints or convenience.

> 翻译: 提供了CPU/GPU 两种版本的 前向/反向

# 损失函数

> [Loss](https://caffe.berkeleyvision.org/tutorial/loss.html)

损失函数在 Caffe 里被定义为了一个 Layer

The final loss in Caffe, then, is computed by summing the total weighted loss over the network, as in the following pseudo-code:

```python
loss := 0
for layer in layers:
  for top, loss_weight in layer.tops, layer.loss_weights:
    # loss_weights: specify their relative importance, 类比损失函数中的\lambda
    loss += loss_weight * sum(top)
```

# Solver

> [Solver](https://caffe.berkeleyvision.org/tutorial/solver.html)

# Interface

1. cmd: 未安装
2. python: 已安装, tutorial 在 examples 文件夹中

