# Caffe 依赖包解析 (P53)

ProtoBuffer: Google 开发的一种实现内存与NVM(非易失存储介质) 交换的协议接口

> 读取参数描述文件(proto), 超参数

Boost

> 在Caffe中使用了Boost的智能指针，避免共享指针时造成内存泄漏或多次释放。
>
> pycaffe使用Boost Python实现C/C++ 与 python的连接

GFLAGS

> 命令行参数解析

GLOG

> 记录应用程序日志的库，提供基于C++标准输入输出流形式的接口
>
> 在caffe中便于开发者查看caffe训练产生的中间输出，跟踪源码，定位

BLAS

> OpenBLAS 矩阵向量计算，是CPU数值计算，该库的性能直接影响caffe的运行性能
>
> GEMM: 矩阵-矩阵乘积运算; GEMV: 矩阵-向量乘积运算

# Examples/MNIST 实战

二进制文件需要转换为 LEVELDB/LMDB 才能被Caffe识别

预处理：

```shell
(caffe_27) caomengqi@ACA-FPGA:~/caffe$ ./examples/mnist/create_mnist.sh
Creating lmdb...
I0728 19:51:58.948565  6355 db_lmdb.cpp:35] Opened lmdb examples/mnist/mnist_train_lmdb
I0728 19:51:58.948951  6355 convert_mnist_data.cpp:88] A total of 60000 items.
I0728 19:51:58.948956  6355 convert_mnist_data.cpp:89] Rows: 28 Cols: 28
I0728 19:52:02.599529  6355 convert_mnist_data.cpp:108] Processed 60000 files.
I0728 19:52:02.884872  6366 db_lmdb.cpp:35] Opened lmdb examples/mnist/mnist_test_lmdb
I0728 19:52:02.885186  6366 convert_mnist_data.cpp:88] A total of 10000 items.
I0728 19:52:02.885192  6366 convert_mnist_data.cpp:89] Rows: 28 Cols: 28
I0728 19:52:03.421725  6366 convert_mnist_data.cpp:108] Processed 10000 files.
Done.
```

LeNet-5 模型：

```protobuf
ame: "LeNet"
layer {
  name: "mnist"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN				// 该层参数只在训练时有效
  }
  transform_param {
    scale: 0.00390625			// 数据变换使用的数据缩放因子
  }
  data_param {
    source: "examples/mnist/mnist_train_lmdb"
    batch_size: 64
    backend: LMDB
  }
}
layer {
  name: "mnist"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TEST					// 该层参数只在测试时有效
  }
  transform_param {
    scale: 0.00390625
  }
  data_param {
    source: "examples/mnist/mnist_test_lmdb"
    batch_size: 100
    backend: LMDB
  }
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"					// 输入blob是data，输出blob是conv1
  param {
    lr_mult: 1					// 权值学习速率倍乘因子
  }
  param {
    lr_mult: 2					// 偏置学习速率倍乘因子，是全局参数的2倍
  }
  convolution_param {
    num_output: 20				// 输出 FM 的数目为20
    kernel_size: 5
    stride: 1
    weight_filler {				// 权值使用Xavier填充器
      type: "xavier"
    }
    bias_filler {				// 偏置使用常数填充器，default=0
      type: "constant"
    }
  }
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX					// Maxpooling
    kernel_size: 2				// 下采样窗口尺寸 2*2
    stride: 2
  }
}
layer {
  name: "conv2"
  type: "Convolution"
  bottom: "pool1"
  top: "conv2"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 50
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "conv2"
  top: "pool2"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "ip1"
  type: "InnerProduct"				// 全连接层
  bottom: "pool2"
  top: "ip1"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  inner_product_param {
    num_output: 500					// output neurons = 500
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "ip1"						// 注意在激活函数中，bottom blob
  top: "ip1"						// 和 top blob 是相同的
}
layer {
  name: "ip2"
  type: "InnerProduct"
  bottom: "ip1"
  top: "ip2"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  inner_product_param {
    num_output: 10					// 添加一个500 dims --> 10 dims的全连接层
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "ip2"
  bottom: "label"
  top: "accuracy"
  include {
    phase: TEST
  }
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "ip2"
  bottom: "label"
  top: "loss"
}
```

Adam Solver

```protobuf
net: "examples/mnist/lenet_train_test.prototxt"

# test_iter specifies how many forward passes the test should carry out.
# In the case of MNIST, we have test batch size 100 and 100 test iterations,
# covering the full 10,000 testing images.
test_iter: 100

# Carry out testing every 500 training iterations.
test_interval: 500

# All parameters are from the cited paper above
base_lr: 0.001
momentum: 0.9
momentum2: 0.999

# since Adam dynamically changes the learning rate, we set the base learning
# rate to a fixed value
lr_policy: "fixed"

# 每100次迭代，在屏幕上打印一次运行log
display: 100

# The maximum number of iterations
max_iter: 10000

# 每5000次迭代打印一次快照，那么如何根据best_accuracy选择合适的时机打印snapshot
snapshot: 5000
snapshot_prefix: "examples/mnist/lenet"

# solver mode: CPU or GPU
type: "Adam"
solver_mode: GPU
```

最后一次快照

```
I0728 20:18:58.591620  6494 solver.cpp:464] Snapshotting to binary proto file examples/mnist/lenet_iter_10000.caffemodel
I0728 20:18:58.597301  6494 sgd_solver.cpp:284] Snapshotting solver state to binary proto file examples/mnist/lenet_iter_10000.solverstate
I0728 20:18:58.600947  6494 solver.cpp:327] Iteration 10000, loss = 0.00324006
I0728 20:18:58.600970  6494 solver.cpp:347] Iteration 10000, Testing net (#0)
I0728 20:18:58.906236  6507 data_layer.cpp:73] Restarting data prefetching from start.
I0728 20:18:58.917526  6494 solver.cpp:414]     Test net output #0: accuracy = 0.9901
I0728 20:18:58.917548  6494 solver.cpp:414]     Test net output #1: loss = 0.0289757 (* 1 = 0.0289757 loss)
I0728 20:18:58.917552  6494 solver.cpp:332] Optimization Done.
I0728 20:18:58.917556  6494 caffe.cpp:250] Optimization Done.
```

### 练习题(P99)

写一个程序将所有LeNet-5识别错误的样本导出

### 解答

```c++
void AccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
const vector<Blob<Dtype>*>& top) {
  ...
  // check if true label is in top k predictions
  for (int k = 0; k < top_k_; k++) {
    if (bottom_data_vector[k].second == label_value) {
      // 预测正确
      ...
    }
    else
    {
      // 预测错误
      // index为batch中的图片序号(0~99)，label为标签值，output为预测值
      LOG(INFO) << "index:" << i << " label:" << label_value << " output:" << bottom_data_vector[k].second;
    }
  }
}
```

这样我们就知道在一个batch中哪些图片被预测错误，以及它的标签值和预测值。测试样本总共有10000个，分为100个batch，每个batch大小为100个，所以我们还需要输出每个batch的序号。跳转到Slover::Test()函数中：

```c++
void Solver<Dtype>::Test(const int test_net_id) {
  ...
  for (int i = 0; i < param_.test_iter(test_net_id); ++i) {
    // 输出batch序号
    LOG(INFO) << "batch:" << i;
  }
}
```

### 解答2(python)

/home/caomengqi/caffe/examples/print_wrong_mnist_samples.py





# 神经网络基础

### 卷积层

卷积层每个样本做一次前向传播时卷积层计算量
$$
Calculations(MAC)=I\cdot J\cdot M\cdot N\cdot K\cdot L
$$
卷积核大小 $I\cdot J$

每个输出通道的特征图大小 $M\cdot N$

$L$ output channels, $K$ input channels

卷积层的学习参数量
$$
Params=I\cdot J\cdot K\cdot L
$$
计算量-参数量之比(Calculations to Parameters Ratio) $= \frac{Calculations}{Params}$

卷积层局部连接特性（相比全连接）也大幅减少了参数量，实现了权值共享。

### 全连接层

主要计算类型是矩阵-向量乘(GEMV)

全连接能的CPR值始终为1，与输入输出维度无关，因此单样本前向传播计算时权值重复利用率很低，但是批样本一次性通过全连接层就会将GEMV升级为GEMM, 提高计算速度。

### 激活函数

激活函数层(layer) 没有权值相关的参数

ReLU函数的源代码：

```c++
#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ReLUForward(
	const int n, 
	const Dtype* in, 
	Dtype* out,
    Dtype negative_slope) 
{
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Forward_gpu(
	const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) 
{
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
  (
      count, 
      bottom_data, 
      top_data, 
      negative_slope
  );
  CUDA_POST_KERNEL_CHECK;
  //     << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void ReLUBackward(
    const int n, 
    const Dtype* in_diff,
    const Dtype* in_data, 
    Dtype* out_diff, 
    Dtype negative_slope) 
{
  CUDA_KERNEL_LOOP(index, n) 
  {
    out_diff[index] = in_diff[index] * ((in_data[index] > 0) + 
                     (in_data[index] <= 0) * negative_slope);
  }
  /* one loop CUDA_KERNEL_LOOP to calculate ReLU 反向 */
  /* negative_slope for leaky ReLU, default = 0 */
  /* ReLU 的反传函数就是: (in_data[idx] > 0)*/
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) 
{
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    // NOLINT_NEXT_LINE(whitespace/operators)
    ReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
    (
        count, 
        top_diff, 
        bottom_data, 
        bottom_diff, 
        negative_slope
    );
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ReLULayer);

}  // namespace caffe
```

sigmoid 反传

sigmoid 函数的导数是$\phi'(x)=\phi(x)\cdot (1-\phi(x))$

```c++
template <typename Dtype>
void SigmoidLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) 
{
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      // top_data 就是前向传播计算的结果 \phi(x), 这里重用降低计算量
      const Dtype sigmoid_x = top_data[i];
      // chain rule: 后一层的误差乘上导函数，即得到前一层的误差
      bottom_diff[i] = top_diff[i] * sigmoid_x * (1. - sigmoid_x);
    }
  }
}
```





# 数据结构(P124)

### 1. Blob

下标访问与 C++ 高维数组一致

##### 本章对blob的成员函数进行了解释

> a.asum_data()		blob.data 的 L1 范数（绝对值之和）
>
> a.asum_diff()		  blob.diff 的 L1 范数
>
> a.sumsq_data()	  L2 范数（平方和）

```c++
#include <vector>
#include <iostream>
#include <caffe/blob.hpp>
using namespace caffe;
using namespace std;
int main(void)
{
  Blob<float> a;
  cout<<"Size : "<< a.shape_string()<<endl;
  a.Reshape(1, 2, 3, 4);
  cout<<"Size : "<< a.shape_string()<<endl;
  float * p = a.mutable_cpu_data();
  float * q = a.mutable_cpu_diff();
  for(int i = 0; i < a.count(); i++)
  {
    p[i] = i;	                     // 将 data 初始化为 1, 2, 3 ..
    q[i] = a.count() - 1 - i;        // 将 diff 初始化为 23, 22, 21, …
  }
  a.Update();	                     // 执行 Update 操作，将 diff 与 data 融合
		                             // 这也是 CNN 权值更新步骤的最终实施者
  for(int u = 0; u < a.num(); u++)
  {
    for(int v = 0; v < a.channels(); v++)
    {
      for(int w = 0; w < a.height(); w++)
      {
        for(int x = 0; x < a.width(); x++)
        {
          cout << "a["<<u<<"]["<<v<<"]["<<w<<"]["<<x<<"] = "
               << a.data_at(u, v, w, x)<<endl;
        }
      }
    }
  }
  cout<<"ASUM = "<<a.asum_data()<<endl;
  cout<<"SUMSQ = "<<a.sumsq_data()<<endl;

  return 0;
}
```



### 2. Layer

```c++
  inline Dtype Forward(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);

  /* The Forward wrapper calls the relevant device wrapper function
   * (Forward_cpu or Forward_gpu) to compute the top blob values given the
   * bottom blobs.  If the layer has any non-zero loss_weights, the wrapper
   * then computes and returns the loss.
   *
   * Your layer should implement Forward_cpu and (optionally) Forward_gpu.
   */

  inline void Backward(
      const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom);

  /**
   * @brief Given the top blob error gradients, compute the bottom blob error
   *        gradients.
   *
   * @param top
   *     the output blobs, whose diff fields store the gradient of the error
   *     with respect to themselves
   * @param propagate_down
   *     a vector with equal length to bottom, with each index indicating
   *     whether to propagate the error gradients down to the bottom blob at
   *     the corresponding index
   * @param bottom
   *     the input blobs, whose diff fields will store the gradient of the error
   *     with respect to themselves after Backward is run
   *
   * The Backward wrapper calls the relevant device wrapper function
   * (Backward_cpu or Backward_gpu) to compute the bottom blob diffs given the
   * top blob diffs.
   *
   * Your layer should implement Backward_cpu and (optionally) Backward_gpu.
   */
```

突然疑问：propagate_down 和 学长们在做的 index flow 岂不是差不多



# Caffe I/O(P170)

DataParameter

数据读取层没有bottom blob，reshape 操作简单

数据读取层反向传播函数不需要做任何操作

DataTransformer

提供了对原始输入图像的预处理方法：随机切块/去均值/灰度色度变换