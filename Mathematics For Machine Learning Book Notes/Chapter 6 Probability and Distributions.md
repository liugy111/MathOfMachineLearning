[TOC]

# 6.1 Construction of a Probability Space

#### 贝叶斯概率和频率派概率

In machine learning and statistics, there are two major interpretations of probability: the Bayesian and frequentist interpretations. 

The Bayesian interpretation uses probability to specify the degree of uncertainty that the user has about an event. It is sometimes referred to as “subjective probability” or “degree of belief”. 

The frequentist interpretation considers the relative frequencies of events of interest to the total number of events that occurred. The probability of an event is defined as the relative frequency of the event in the limit when one has infinite data.

频率学派和贝叶斯学派最大的差别其实产生于对参数空间的认知上。

所谓参数空间，就是你关心的那个参数可能的取值范围。

频率学派（其实就是当年的Fisher）并不关心参数空间的所有细节，他们相信数据都是在这个空间里的”某个“参数值下产生的（虽然你不知道那个值是啥），所以他们的方法论一开始就是从“哪个值最有可能是真实值”这个角度出发的。于是就有了最大似然（maximum likelihood）以及置信区间（confidence interval）这样的东西，你从名字就可以看出来他们关心的就是我有多大把握去圈出那个唯一的真实参数。

而贝叶斯学派恰恰相反，他们关心参数空间里的每一个值，因为他们觉得我们又没有上帝视角，怎么可能知道哪个值是真的呢？所以参数空间里的每个值都有可能是真实模型使用的值，区别只是概率不同而已。于是他们才会引入先验分布（prior distribution）和后验分布（posterior distribution）这样的概念来设法找出参数空间上的每个值的概率。

#### 概率空间 Probability Space 

