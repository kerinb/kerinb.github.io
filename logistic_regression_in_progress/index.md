---
layout: post
title: "Logistic Regression from Scratch in Python "
date: Monday 14th January 2020
---

<meta charset="UTF-8">
<meta name="description" content="Logistic Regression tutorial in python from scratch">
<meta name="keywords" content="Python, Machine Learning, Logisitc Regression, Tutorial, Tutorial from scratch">
<meta name="author" content="Breandán Kerin">

<p>
	<b>Requirements for this tutorial: </b>
	<ol>
		<li> 
			Python
		</li>
		<li> 
			NumPy
		</li>
		<li> 
			Pandas
		</li>
		<li> 
			Sci-kit Learn (sklearn)
		</li>
	</ol>
</P>

<p>
	<b> Post Summary: </b>  This tutorial covers the topic of Logistic Regression (LogReg) using gradient 
	descent. LogReg an example of a machine learning classification algorithm, where, instead of trying to
	predict a continuous value	as was done in Linear Regression, LogReg aims to classify teh data for 
	example, is the image a cat or a dog?. This tutorial will walk through an implementation of LogReg 
	from scratch written in Python. This tutorial assumes some basic knowledge of Gradient Descent. IF you 
	are new to ML or simply need a refersher, check out my tutorial on 
	<a href="/blog/2020/01/14/linear-regression-from-scratch">Linear Regression using Gradient Descent</a>.
</p>

<p>
	<b>Note:</b> While im in the process of setting up a comments section on the tutorial/blog pages, 
	please feel free to provide feed back via my <em>email:
	</em><a href="mailto:b.m.kerin96@gmail.com">b.m.kerin96@gmail.com</a>
</p>

<h2> 
	I just came here for the code:
</h2>

<figure>
<pre class="brush: python">
<code>
1. import numpy as np
2. import pandas as pd
3. import matplotlib.pyplot as plt
4. from sklearn.datasets import load_iris
5. from sklearn import datasets
6. from sklearn.model_selection import train_test_split
7. import math
8. 
9. def sigmoid(x):
10.     return 1 / (1 + np.exp(-x))
11.
12. def loss_function(y_train, prediction):
13.     return (1/len(y_train))*np.sum(y_train[0] * np.log(prediction).transpose() + (1-y_train[0])*np.log(1-prediction))
14.
15. def add_theta_0(x, X):
16.     x["INTERCEPT"] = pd.Series(np.ones(X.shape[0]))
17.     return x
18.
19. def normalise(X): 
20.     min_val = np.min(X, axis = 0) 
21.     max_val = np.max(X, axis = 0) 
22.     range_vals = max_val - min_val 
23.     norm_X = 1 - ((max_val - X)/range_vals) 
24.     return norm_X 
25. 	
26. r=0.77; s=1; min_error = 10e-10
27. num_iters = 10000; alpha = 1.0
28. 
29. iris = datasets.load_iris() 
30.
31. X = normalise(pd.DataFrame(iris.data[0:99, :2]))
32. X = add_theta_0(X, X)
33. y = pd.DataFrame(iris.target[0:99])
34. 
35. X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
36. 
37. init_weights = np.zeros((len(X_train.columns)))
38. weights_list = [init_weights]
39. weights = init_weights
40. losses = []
41. prev = math.inf
42. 
43. for n_iter in range(num_iters):
44.     z = np.dot(weights.T, X_train.T)
45.     prediction = sigmoid(z)
46.     loss =- loss_function(y_train, prediction)
47.     derivative = (1/len(X_train)) * np.dot(X_train.T, (prediction - y_train[0]).transpose())
48.     weights -= alpha * derivative
49.     losses.append(loss)
50. 
51. Epoch=pd.DataFrame(list(range(100,100001,100)))
52. loss=pd.DataFrame(losses)
53. loss_data=pd.concat([Epoch, loss], axis=1)
54. loss_data.columns=['Epoch','loss']
55. plt.scatter(loss_data['Epoch'], loss_data['loss'])
56. plt.xlabel('Epoch')
57. plt.ylabel('loss')
</code>
</pre>
</figure>

<p>
	The code above can also be found on my github 
	<a href='https://github.com/kerinb/CodeBlogRepo/tree/master/LogisticRegressionBlog'>here</a>.
</p>

<h2>
	Section 1: Some Preliminary Theory: 
</h2>

<p>
	This tutorial will discuss the basics of Logistic Regression from its prediction function and loss 
	function to a simple implementation in python using only Pandas and Numpy. At the tend, the results of
	the 'from scratch' implementation will be compared to the results obtained from sklearns implementation.
	Logistic Regression (LogReg) is a supervised classified machine learning algorithm. The purpose of a 
	classification algorithm is to take input data and to map it onto a set of known labels. For example, 
	if we had a collection of images and each image contained either a snapshot of a cat or dog. We would 
	want Logistic Regression to correctly classify an image as either a cat or a dog. 
</p>

<p>
	Similar to the Linear Regression tutorial posted last week, there are two variables that we are 
	concerned with in LogReg and these are the independent variable, <i>X</i> and the dependent variable 
	<i>y</i>. Where <i>X</i> is our observed data (In the image example, this would correspond to the 
	values of the pixels in our images) and <i>y</i> is our target, or labels that we want to correctly 
	predict. 
</p>

<p>
	In classification problems, there are generally many different types of labels. For example, if we
	are classifying images we may want to classify the image as a cat, a dog, a bird, or an elephant. 
	However, for this implementation of LogReg, we assume that we only have two labels for simplicity. 
	So, for this tutorial we can take the prediction as a probabilty which will lie between [0, 1]. As
	a general rule of thumb, if the probability is greater than or equal to 0.5, then we can label this 
	as class 1, otherwise, we will label it as class 0. 
</p>

<p>
	The difference between Linear Regression and LogReg is with our prediction and loss functions. 
	For Linear Regression we want to make a prediction using a function similar to:<br>
	<center><i>h_θ (x) = θ * x</i></center> <br>
	And we will use the Mean Square Loss function as our loss function:<br>
	<center> <i> error = ∑_(i=1)^N ( prediction - y)^2 </i> </center><br>
	However, for the LogReg, we will use the sigmoid function as our prediction value. This will give us
	the desired range of 0 to 1. The output of the sigmoid function can be seen in the below image:
</p>

![](/files/LogisticRegressionBlog/logRegRange.png)

<p>
	With the above image, we can notice several features of the sigmiod function:<br>
	1. For large values of input into output tends towards 1<br>
	2. For negative inputs into the sigmoid, the output tends towards 0. <br>
	3. Sigmoid(0) = 0.5<br>
	4. Sigmooid(input) >= 0.5, input >= 0<br>
	5. Sigmoid(input) < 0.5 input < 0<br>
</p>

<p>
	When mentioned above that the rule of thumb is we will use 0.5 to determine whether we will classify
	the result as class 1 or class 0 - The value, 0.5, is known as a <i>Decision Boundary</i> since it
	is used to create a boundary for classification by the model. <br>
	<center><i> if h_θ (X.T*θ.T) >= 0.5, then y  = 1 </i></center><br>
	<center><i> if h_θ (X.T*θ.T) < 0.5, then y  = 0 </i></center><br> 
	
	Therefore, we can see the decision boundary is a line defined by <i> X.T*θ.T </i> since it seperates the 
	the area where y = 0 and y = 1. It should be noted that the <i>line</i> here can be <i>non-linear</i> 
	since its defined by the feature variables <i>X</i> which can be non-linear. 
</p>

<p>
	In order to compute the loss in LogReg, we use a loss function that's derived from the sigmoid function
	which helps us to find <i>θ</i> that will define the optimal decision boundary <i> X.T*θ.T </i>. Once
	we find the optimal values of <i>θ</i> , the prediction <i>h_θ (X.T*θ.T)</i>, will tell us what side 
	of the decision boundary our prediction will lie on. 
</p>

<h2>
	Section 2: Putting Theory into practice: 
</h2>

<p>
	FIND A DATASET AND TALK ABOUT IT ETC...
</p>

<figure>
<pre class="brush: python">
<code>
1. ENTER CODE HERE WITH COMMENTS!!!
</code>
</pre>
</figure>

<p>
	DISCUSS CODE AND / OR OPERATIONS USED!! 
</p>

<figure>
<pre class="brush: python">
<code>
1. ENTER SKLEARN CODE HERE WITH COMMENTS!!!
</code>
</pre>
</figure>