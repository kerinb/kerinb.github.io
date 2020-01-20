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
		<li> 
			MatPlotLib
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
	For Linear Regression we want to make a prediction using a function similar to:
</p>

<p>
	<center><i>h_θ (x) = θ * x</i></center> <br>
</p>

<p>
	And we will use the Mean Square Loss function as our loss function:<br>
</p>

<p>
	<center> <i> error = ∑_(i=1)^N ( prediction - y)^2 </i> </center><br>
</p>

<p>
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
	The sigmoid function can be re-written as follows:
</p>

<p>
	<center><i>sigmoid(X) = exp(X)/(exp(X) + exp(0))</i></center>
</p>

<p>
	The derivative of the sigmoid fuction is:
</p>

<p>
	<center><i>delta sigmoid(X) = sigmoid(X) * (1 - sigmoid(X))</i></center>
</p>

<p>
	When mentioned above that the rule of thumb is we will use 0.5 to determine whether we will classify
	the result as class 1 or class 0 - The value, 0.5, is known as a <i>Decision Boundary</i> since it
	is used to create a boundary for classification by the model.
</p>

<p>
	<center><i> if h_θ (X.T*θ.T) >= 0.5, then y  = 1 </i></center><br>
	<center><i> if h_θ (X.T*θ.T) < 0.5, then y  = 0 </i></center><br> 
</p>

<p>
	Therefore, we can see the decision boundary is a line defined by <i> X.T*θ.T </i> 
	since it seperates the 
	the area where y = 0 and y = 1. It should be noted that the <i>line</i> here can be <i>non-linear</i> 
	since its defined by the feature variables <i>X</i> which can be non-linear. 
</p>

<p>
	In order to compute the loss in LogReg, we use a loss function that's derived from the sigmoid function
	which helps us to find <i>θ</i> that will define the optimal decision boundary <i> X.T*θ.T </i>. 
	Once we find the optimal values of <i>θ</i> , the prediction <i>h_θ (X.T*θ.T)</i>, 
	will tell us what side of the decision boundary our prediction will lie on. 
</p>

<p>
	The loss function we use in LogReg is called the Maximum Log Likelihood (MLE) function and is defined as:
</p>

<p>
	L_θ(X) = 1/N * SUM(y * Log(sigmoid(X).T) + (1 - y) * log(1 - sigmoid(X))
</p>

<p>
	This function can be derived easily. Take a single sample from the entire population and this 
	record will follow a Bernoulli distribution:
</p>

<p>
	<i><center>p(y|X) = P^y(1 - p)^(1 - y)</center></i>
</p>

<p>
	In the above formula, <i>y</i> is an indicator of class, either a 1 or 0 and p is the probability 
	that the event occurs. If instead of a single record, we had N many records in the datasets we would
	then need to alter the original function above such that it can handle the N many records. If we assume
	that each record in te dataset is independent and identically distributed, we can compute the probability
	as:
</p>

<p>
	<i><center>P = P(y_1, X_1) * P(y_2, X_2) * ... * P(y_N, X_N)</i></center><br>
	<i><center>P = PRODUCT(p^y_i(1-p)^)(1-y_i))</i></center>
</p>

<p>
	Now, we take the log of both sides of the equation so we can calculate the log likelihood
</p>

<p>
	<i><center>log(P) = log(PRODUCT(p^y_i(1-p)^)(1-y_i)))</i></center>
	<i><center>log(P) = SUM(y_i * log(P) + (1 - y_i) * log(1 - P)</i></center>
</p>

<p>
	Note that, in the above equation, the probability P is calculated using the sigmoid function. The 
	MLE is often used to obtain the parameter for a distribution. With this, when we maximise the log 
	likelihood, we solve the dual problem of minimising the cost function. <br>
	<b>Note:</b> I currently do not have any tutorials or notes published relating to convex optimisation,
	and thus related to what the dual is - This wiki 
	<a href="https://en.wikipedia.org/wiki/Duality_(optimization)">link</a> provides a good explanation.
	<br>
	The loss function we use in LogReg is provided below:
</p>

<p>
	<i><center>J(X, θ) = (1 / N) * SUM(y_i * log(sigmoid(X, θ)) + (1- y_i) * log(1 - sigmoid(X, θ))</i></center>
</p>

<p>
	Now that we know how to compute the loss with the negative log likelihood, the next step we need to do
	is to compute the derivative of this function and apply it to gradient descent so that we find the 
	optimal values of θ such that it minimises the loss function. I wont dwell to long on deriving the 
	derivitive of the loss function. The derivtive simplifies to:
</p>

<p>
	<i><center>delta J(X, θ) = 1/N * SUM((sigmoid(X_i, θ) - y_i)X_i)</i></center>
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
1. import numpy as np
2. import pandas as pd
3. import matplotlib.pyplot as plt
4. from sklearn.datasets import load_iris
5. from sklearn import datasets
6. from sklearn.model_selection import train_test_split
7. from sklearn import linear_model
8. import math
9.
10. def sigmoid(x):
11.     return 1 / (1 + np.exp(-x))
12. 
13. def loss_function(y_train, prediction):
14.     return (1/len(y_train))*np.sum(y_train[0] * np.log(prediction).transpose() + (1-y_train[0])*np.log(1-prediction))
15. 
16. def add_theta_0(x, X):
17.     x["INTERCEPT"] = pd.Series(np.ones(X.shape[0]))
18.     return x
19. 
20. def normalise(X): 
21.     min_val = np.min(X, axis = 0) 
22.     max_val = np.max(X, axis = 0) 
23.     range_vals = max_val - min_val 
24.     norm_X = 1 - ((max_val - X)/range_vals) 
25.     return norm_X 
26.	
27. # required constants for script
28. r=0.77; s=1; min_error = 10e-10
29. num_iters = 10000; alpha = 1.0
30. 
31. iris = datasets.load_iris() # I may end up having to drop one of the labels
32. # We will see if I can ge multi-classification to work! 
33. 
34. X = normalise(pd.DataFrame(iris.data[0:99, :2]))
35. X = add_theta_0(X, X)
36. y = pd.DataFrame(iris.target[0:99])
37.
37. X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
39.
40. #### START LOGISTIC REGRESSION ####
41. # Define variables to store weights and losses
42. init_weights = np.zeros((len(X_train.columns)))
43. weights_list = [init_weights]
44. weights = init_weights
45. losses = []
46. prev = math.inf
47.
48. for n_iter in range(num_iters):
49.     z = np.dot(weights.T, X_train.T)
50.     prediction = sigmoid(z)
51.     loss =- loss_function(y_train, prediction)
52.     derivative = (1/len(X_train)) * np.dot(X_train.T, (prediction - y_train[0]).transpose())
53.     weights -= alpha * derivative
54.     losses.append(loss)
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