# Reference: https://deshanadesai.github.io/notes/Fancy-PCA-with-Scikit-Image


import numpy as np
import matplotlib.pyplot as plt
from skimage import io,transform

# Using six sample images.
imnames = ['n00.jpg','n01.jpg','n02.jpg','n03.jpg','n04.jpg','n05.jpg']


# Read collection of images with imread_collection
imlist = (io.imread_collection(imnames))


# initializing with zeros. urn the image matrix of m x n x 3 to lists of rgb values i.e. (m*n) x 3.
res = np.zeros(shape=(1,3))


# 对于第一张图片:
m=transform.resize(imlist[0],(256,256,3))
# Reshape the matrix to a list of rgb values.
arr=m.reshape((256*256),3)
# concatenate the vectors for every image with the existing list.
res = np.concatenate((res,arr),axis=0)


# delete initial zeros' row
res = np.delete(res, (0), axis=0)


# Subtract the mean. 
# For PCA to work properly, you must subtract the mean from each of the dimensions.
m = res.mean(axis = 0)
res = res - m


"""
Output(res):
[[ 0.14435945 0.15325546 0.06625693]
[ 0.13651632 0.14933389 0.05057065]
[ 0.13259475 0.1473731 0.03096281]
...,
[-0.19991138 -0.34395655 -0.37997469]
[-0.1827392 -0.32339895 -0.35941709]
[-0.13852964 -0.24389528 -0.29803535]]
"""


# Calculate the covariance matrix. 
# Since the data is 3 dimensional, the cov matrix will be 3x3. No surprises there.
R = np.cov(res, rowvar=False)

"""
Output (R):
[[ 0.06814677 0.06196288 0.05043152]
[ 0.06196288 0.06608583 0.06171048]
[ 0.05043152 0.06171048 0.07004448]]
"""

# Calculate the Eigenvectors and Eigenvalues of the covariance matrix.
from numpy import linalg as LA

evals, evecs = LA.eigh(R)
idx = np.argsort(evals)[::-1]
evecs = evecs[:,idx]
evals = evals[idx] # sort eigenvectors according to same index
evecs = evecs[:, :3] # select the best 3 eigenvectors (3 is desired dimension of rescaled data array)
evecs_mat = np.column_stack((evecs)) # make a matrix with the three eigenvectors as its columns.


# carry out the transformation on the data using eigenvectors
# and return the re-scaled data, eigenvalues, and eigenvectors
m = np.dot(evecs.T, res.T).T


"""
having performed PCA on the pixel values, 
we need to add multiples of the found principal components 
with magnitudes proportional to the corresponding eigenvalues 
times a random variable drawn from a Gaussian distribution 
with mean zero and standard deviation 0.1. 
"""

def data_aug(img = img):
	mu = 0
	sigma = 0.1
	feature_vec=np.matrix(evecs_mat)

	# 3 x 1 scaled eigenvalue matrix
	se = np.zeros((3,1))
	se[0][0] = np.random.normal(mu, sigma)*evals[0]
	se[1][0] = np.random.normal(mu, sigma)*evals[1]
	se[2][0] = np.random.normal(mu, sigma)*evals[2]
	se = np.matrix(se)
	val = feature_vec*se

	# Parse through every pixel value.
	for i in xrange(img.shape[0]):
		for j in xrange(img.shape[1]):
			# Parse through every dimension.
			for k in xrange(img.shape[2]):
				img[i,j,k] = float(img[i,j,k]) + float(val[k])


img = imlist[0]/255.0 # perturbing color in image[0], re-scaling from 0-1
data_aug(img)
plt.imshow(img)
