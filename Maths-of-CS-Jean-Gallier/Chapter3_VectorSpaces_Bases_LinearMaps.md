# 3.1 向量空间

对于向量空间 $E$ 定义加法(vector addition) 和放缩(scalar multiplication, element-wise) 

```
(V0) E is an abelian group w.r.t. +, with identity element 0;
(V1) α · (u + v) = (α · u) + (α · v);
(V2) (α + β) · u = (α · u) + (β · u);
(V3) (α ∗ β) · u = α · (β · u); (其中 ∗ 代表数乘)
(V4) 1 · u = u
```

One may wonder whether axiom (V4) is really needed. Could it be derived from
the other axioms? The answer is no. For example, one can take $E = R^n$ and define $R \times R^n → R^n$ by
$$
\lambda \cdot (x_1 ,...,x_n) = (0,...,0)
$$
for all $(x_1 ,...,x_n ) \in R^n$ and all $\lambda \in R$. Axioms (V0)–(V3) are all satisfied, but (V4) fails.

# 3.2 Indexed Families
Indexed Families: 顾名思义就是序列的索引

the Sum Notation $\sum_{i \in I}a_i$ 的引入：我们已经有了二元的加法运算符(a1 +a2)，用 $\sum$ 定义连加

# 3.3 线性独立，子空间

The vector space of real polynomials, $R[X]$, does not have a finite basis but instead it has an infinite basis, namely $\{1, X, X^2 , ...,X^n , ...\}$

All bases of a vector space have the same number of elements (cardinality), which is called
the dimension of the space.

在线性代数里，矢量空间的一组元素中，若没有矢量可用有限个其他矢量的线性组合所表示，则称为线性无关或线性独立 (linearly independent)，反之称为线性相关(linearly dependent)。

TODO：P55 Definition 3.3