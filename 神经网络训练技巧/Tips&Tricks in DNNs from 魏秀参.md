[TOC]

# 1. data augmentation

<img src="http://p3.pstatp.com/large/pgc-image/1527060447020a2a66c9196" style="zoom:50%;" />

## 1.1 Horizontal/Vertical Flip：水平/垂直翻转

## 1.2 Random Crops：随机裁剪

## 1.3 Color Jittering (HSV Color Space)

> From: wiki

色相（H, Hue）是色彩的基本属性，就是平常所说的颜色名称，如红色、黄色等。饱和度（S, Saturation）是指色彩的纯度，越高色彩越纯，低则逐渐变灰，取0-100%的数值。明度（V, Value, aka Brightness），亮度（L），取0-100%。

> From: 魏秀参

1. raise saturation and value (S and V components of the HSV color space) of all pixels to a power between 0.25 and 4 (same for all pixels within a patch), multiply these values by a factor between 0.7 and 1.4, and add to them a value between -0.1 and 0.1

2. add a value between [-0.1, 0.1] to the hue (H component of HSV) of all pixels in the image/patch. 

## 1.4 Shift：平移变换

## 1.5 Rotation：旋转

## 1.6 Noise：高斯噪声、模糊处理

## 1.7 Fancy PCA

> From: 魏秀参

Fancy PCA alters the intensities of the RGB channels in training images. In practice, you can firstly perform PCA on the set of RGB pixel values throughout your training images.

> From: AlexNet Paper

Specifically, we perform PCA on the set of RGB pixel values throughout the ImageNet training set. To each training image, we add multiples of the found principal components, with magnitudes proportional to the corresponding eigenvalues times a random variable drawn from a Gaussian with mean zero and standard deviation 0.1. Therefore to each RGB image pixel
$$
I_{xy} = \left [ I^R_{xy}, I^G_{xy}, I^B_{xy} \right ]^T
$$
 we add the following quantity:
$$
\left [ \mathbf p_1, \mathbf p_2, \mathbf p_3 \right ] \left [ \alpha_1 \lambda_1, \alpha_2 \lambda_2, \alpha_3 \lambda_3 \right ]^T
$$
 p 和 \lamda 分别代表第i个特征向量, RGB像素值3x3协方差矩阵的特征值。

原论文 claimed that “fancy PCA could approximately capture an important property of natural images, namely, that **object identity is invariant to changes in the intensity and color of the illumination**”. To the classification performance, this scheme reduced the top-1 error rate by over 1% in the competition of ImageNet 2012.

Fancy PCA Implementation: see [[my python code here]](./fancypca.py)

# 2. pre-processing on images

Please note that, we describe these pre-processing here just for completeness. In practice, these transformations are not used with Convolutional Neural Networks. However, it is also very important to zero-center the data, and it is common to see normalization of every pixel as well. 

## 2.1 Normalization

```python
# normalizes each dimension so that 
# the min and max along the dimension is -1 and 1 respectively
X -= np.mean(X, axis = 0) # zero-center
X /= np.std(X, axis = 0) # normalize
```

It only makes sense to apply this pre-processing if you have a reason to believe that different input features have different scales (or units), but they should be of approximately equal importance to the learning algorithm. In case of images, the relative scales of pixels are already approximately equal (and in range from 0 to 255), so it is not strictly necessary to perform this additional pre-processing step. 

## 2.2 PCA Whitening

```python
X -= np.mean(X, axis = 0) # zero-center
cov = np.dot(X.T, X) / X.shape[0] # compute the covariance matrix
U,S,V = np.linalg.svd(cov) # compute the SVD factorization of the data covariance matrix
Xrot = np.dot(X, U) # decorrelate the data
Xwhite = Xrot / np.sqrt(S + 1e-5) # divide by the eigenvalues (which are square roots of the singular values)
```

Note that here it adds 1e-5 (or a small constant) to prevent division by zero. One weakness of this transformation is that it can greatly exaggerate the noise in the data, since it stretches all dimensions (including the irrelevant dimensions of tiny variance that are mostly noise) to be of equal size in the input. This can in practice be mitigated by stronger smoothing (i.e., increasing 1e-5 to be a larger number). 

# 3. Initializations of Networks



# some tips during training





# selections of activation functions



# diverse regularizations



# some insights found from figures



# methods of ensemble multiple deep networks

