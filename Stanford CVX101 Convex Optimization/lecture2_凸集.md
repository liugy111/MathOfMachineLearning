[Today's Bilibili Link](https://lagunita.stanford.edu/courses/Engineering/CVX101/Winter2014/about)

# Dry Conception

Affine set: contains the line through any two distinct points in the set

Convex set: Affine set with $0 \leq\theta\leq 1$

Convex hull: set of all convex combinations

Convex Cone: set that contains all conic (non-negative) combinations of points in the set

Hyperplane: $\{x|a^Tx=b\}$, $a$ 是法向量 (normal vector)

> 证明 halfspace 半空间 是凸的
> $$
> a^Tx_1 \leq b\\
> a^Tx_2 \leq b\\
> x = \theta x_1 + (1-\theta) x_2 \leq b \text{ as well}.
> $$

Euclid Ball $\{x |  (x-x_c)^T(x-x_c) \leq r^2\}$是凸集

##### Ellipsoid

Ellipsoid 椭球 $\{x |  (x-x_c)^TP^{-1}(x-x_c) \leq 1\}$是欧几里得球的泛化形式

$P$ symmetric positive definite

If $P=r^2I$, ellipsoid ----> Euclid ball

Ellipsoid 的其他表示方法: $\{x_c+Au, \text{| } ||u||_2 \leq 1\}$

- linear transformation of $u$, plus a bias $x_c$

- when A square and nonsingular, A is not unique 
  - for example, $AQ$ can substitude $A$ if $Q$ is orthogonal (that is, $Q^TQ = I$)
- when A symmetric positive definite, A unique (proved by SVD)

##### Norm balls $\{x \text{ | } ||x-x_c|| \leq r\}$

norm function 的三个条件:

- positive and definiteness
- $||tx||=|t|||x||$
- 范数的三角不等式

##### Polyhedra / Polytopes

solution set of finitely many linear inequalities and equalities

polyhedron is intersection of finite number of halfspaces (inequalities) and hyperplanes (equalities)

##### Positive Semidefinite Cone

$$
\text{A symmetric square matrix } X \in S^n_+ \Leftrightarrow z^TXz \geq 0 \text{ for all }z
$$

# Operations (calculus) that preserve convexity

1. the intersection of (any number of) convex sets is convex

2. 仿射函数 affine functions

   > scaling, translation (平移/旋转), projection
   >
   > the image of a convex set under $f$ is convex
   >
   > the inverse image $f^{−1} (C)$ of a convex set under $f$ is convex
   >
   > example:  solution set of linear matrix inequality $\{x|x_1A_1+...+x_mA_m \leq B\}$

3. perspective function / projective mapping

   透视函数对向量进行伸缩，或称为规范化，使得最后一维分量为1并舍弃之

   images and inverse images of convex sets under linear-fractional functions
   are convex

   > $R^{n+1} \rightarrow R^n$, $P(x,t) = x/t$

4. linear-fractional functions

   images and inverse images of convex sets under perspective are convex

   > $f(x)=\frac{Ax+b}{c^Tx+d}$

# 广义不等式

proper cone: convex, closed/ solid/ pointed

example:  

positive semidefinite cone $S_+^n$: positive semidefinite n × n matrices

nonnegative polynomials on [0,1]: $K=\{x\in R^n | x_1+x_2t+...+x_nt^{n-1} \geq 0 \text{ for } t \in [0,1]\}$