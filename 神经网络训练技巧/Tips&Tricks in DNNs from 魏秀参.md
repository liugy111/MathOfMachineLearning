[TOC]

# 1. data augmentation

## 1.1 Horizontal/Vertical Flip：水平/垂直翻转

## 1.2 Random Crops

## 1.3 Color Jittering (HSV Color Space)

> From: wiki

色相（H, Hue）是色彩的基本属性，就是平常所说的颜色名称，如红色、黄色等。饱和度（S, Saturation）是指色彩的纯度，越高色彩越纯，低则逐渐变灰，取0-100%的数值。明度（V, Value, aka Brightness），亮度（L），取0-100%。

> From: 魏秀参

1. raise saturation and value (S and V components of the HSV color space) of all pixels to a power between 0.25 and 4 (same for all pixels within a patch), multiply these values by a factor between 0.7 and 1.4, and add to them a value between -0.1 and 0.1

2. add a value between [-0.1, 0.1] to the hue (H component of HSV) of all pixels in the image/patch. 

## 1.4 Shift：平移变换

## 1.5 Rotation/Reflection：旋转/仿射变换

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

# pre-processing on images



# initializations of Networks



# some tips during training





# selections of activation functions



# diverse regularizations



# some insights found from figures



# methods of ensemble multiple deep networks

