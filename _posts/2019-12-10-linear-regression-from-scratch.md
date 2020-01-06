---
layout: post
title: "Linear Regression "
date: 2019-12-10
---

<h1> 
	Linear Regression from Scratch - Python
</h1>

<p>
	Requirements for this tutorial:
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
	<b>Post Summary:</b> This tutorial covers the topic of Linear Regression (LR), a simple and very well understood machine
	learning algorithm. This tutorial will walk through an implementation of LR from scratch written in Python. LR is a 
	perfect yet simple example of supervised learning - a branch of machine learning that learns from labelled data with
	known true values.
</p>

<p>
	<h3>I just came here for the code:</h3> <br>
    <figure>
        <pre class="brush: python">
            <code>
                init_weights = np.zeros(len(X_train.columns))
                weights_list = [init_weights]
                losses = []
                weights = init_weights
                prev = math.inf
                for n_iter in range(num_iters):
                    err = y_train - X_train.dot(weights)
                    grad = -X_train.T.dot(err) / len(err)
                    loss = np.sqrt(2 * 1/2*np.mean(err**2))            
                    weights = weights - gamma * grad
                    weights_list.append(weights)
                    losses.append(loss)
                    if(abs(loss - prev) < epsilon) :
                        print("Reached Convergence !")
                        break
                    prev = loss
                params = weights_list[-1]  
            </code>
        </pre>
    </figure>
</p>

<h3>Section 1: Some Preliminary Theory:</h3>

<p>
	LR is a statistical model that observes the linear relationships between a dependent variable and a set of independent 
	varables. In LR, we aim to model our data in the following manner: 
	<!-- INSERT IMAGE OF y = theta_0 + theta_1 * x-->
	Where <bold>y</bold> is the dependent variable that we want to predict. <bold>X</bold> is the inpdendent variable. 
	<bold>theta_0</bold> and <bold>theta_1</bold> are the weights we want to estimate in order to predict <bold>y</bold>.	
	The number of weights we have to learn is equal to the number of dependent variables from the dataset we are using.
	For example, if we input a dataset with 10 columns, we will have 10 weights that must be learned in 
	order to successfully predict out <bold>y</bold> value. 
</p> 

<p>
	We know that the variable <bold>theta_0</bold> corresponds to the intercept of our estimated line with the 
	y-axis. In machine learning, this is generally called the bias, due to the fact that it is added to offset all the
	predictions that are made from the x-axis. the <bold>theta</bold> values are the slope of the line, since it defines 
	the slope of the line. 
</p>

<p> 
	We can now see that the ultimate goal of LR is to find the best estimates for the weights that can be used to  
	estimate the dependent variable <bold>y</bold>. We do this by trying to minimise the errors we obtain when predicting
	the <bold>y</bold> values. In a more mathematically regorous definition we are trying to achieve the following:
	<!-- INSERT IMAGE OF LR maths -->
	This translates into: We want to find the value for the weights, theta, that minimise the error between our 
	prediction and the actual value. 
</p>

<p>
    MAYBE ADD MORE INFORMATION ON MATHS BEHIND THIS?
</p>

<h3>Section 2: Putting theory into practice:</h3>

<p>
	Now, imagine that you are considering moving to Boston, MA, and you want to purchase a house. If you, for some 
	strange reason, do not have access to a real estate agent, but have access to a datset relating to the prices of 
	houses in Boston and you want to know what sort of house fits within your budget, you could use this to estimate 
	the cost of your new home! 
</p>

<!-- ADD CODE WITH COMMENTS HERE -->
<figure>
<pre class="brush: python">
<code>

    # Imports
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_boston
    from sklearn.metrics import r2_score
    import math
    
    # required constants for script
    r=0.77; s=1; epsilon = 10e-10
    num_iters = 10000; gamma = 0.01
    
    # Load data 
    boston = load_boston()
    boston_dataset = pd.DataFrame(boston.data, columns=boston.feature_names)
    boston_dataset['MEDV'] =  boston.target
    y = boston_dataset.MEDV
    X = boston_dataset.drop(['MEDV'], axis=1)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y, r, seed=s)
    X_train = add_theta_0((X_train - X_train.mean(axis=0))/ X_train.std(axis=0), X)
    X_test = add_theta_0((X_test - X_test.mean(axis=0))/ X_test.std(axis=0), X)
    
    #### START LINEAR REGRESSION ####
    # Define variables to store weights and losses
    init_weights = np.zeros(len(X_train.columns))
    weights_list = [init_weights]
    weights = init_weights
    losses = []
    prev = math.inf
    
    for n_iter in range(num_iters):
        # compute loss, gradient and rmse(actual loss)
        err = y_train - X_train.dot(weights)
        grad = -X_train.T.dot(err) / len(err)
        loss = np.sqrt(2 * 1/2*np.mean(err**2))
        
        # gradient w by descent update
        weights = weights - gamma * grad
        
        # store w and loss
        weights_list.append(weights)
        losses.append(loss)
        
        #Stop earlier if we reached convergence
        if(abs(loss - prev) < epsilon) :
            print("Reached Convergence !")
            break
        prev = loss
        
    #Get final weights
    params = weights_list[-1]    

</code>
</pre>
</figure>

<p>
	There were a couple of helper functions that I also implemented to i) split the data into train and test sets, ii)
	to standardise the data and iii) evaluate the implementation using the R squared measure. The code for these are 
	shown below:
</p>

<figure>
<pre class="brush: python">
<code>

    def split_data(x, y, ratio, seed=1):
        """split the dataset based on the split ratio."""
        # set seed to produce reproducable results
        np.random.seed(seed)
        
        # generate random indices
        indices = np.random.permutation(len(y))
        index_split = int(np.floor(ratio * len(y)))
        
        # create split
        x_train = x.iloc[indices[: index_split]]
        x_test = x.iloc[indices[index_split:]]
        y_train = y.iloc[indices[: index_split]]
        y_test = y.iloc[indices[index_split:]]
        return x_train, x_test, y_train, y_test
    
    def add_theta_0(x, X):
        x["INTERCEPT"] = pd.Series(np.ones(X.shape[0]))
        return x
        
</code>
</pre>
</figure>


<table style="width:100%">
  <tr>
    <th>Operation</th>
    <th>Explained</th>
  </tr>
  <tr>
    <td>X</td>
    <td>Input matrix of independent variables: Each row is a unique training sample</td>
  </tr>
  <tr>
    <td>y</td>
    <td>Output matrix of dependent varables: Each row is unique training sample</td>
  </tr>
  <tr>
    <td>*</td>
    <td>element wise multiplication: Two vectors of equal dimensions are multiplied together to produce a resultant 
    matrix of the same dimension as the two original matrices.</td>
  </tr>
  <tr>
    <td>-</td>
    <td>element wise subtraction: Two vectors of equal dimensions are subtracted frome each other to produce a resultant
    matrix of the same dimension as the two original matrices</td>
  </tr>
  <tr>
    <td>x.dot(y)</td>
    <td>If x and y are vectors this is the dot product, if both x and y are matrices, this is matrix multiplication and 
    if either or y is a vector or a matrix, this is a vector-matrix multipication</td>
  </tr>
  <tr>
    <td>add_theta_0</td>
    <td>This is a helper function that was added to append the INTERCEPT column to the independent variables. This is 
    done to simplify the matrix multiplication from y = theta_0 + x*remaining_thetas to y = theta*x</td>
  </tr>
  <tr>
    <td>R Squared</td>
    <td>R Squares is a statistical measure that ranges from 0-100% and is used to illustrate the amount of explained 
    variance in the learned model. The formula for R squared is: R^2 = 1 - explained variation/total variation</td>
  </tr>
</table>

<p>
    As you can see from the 'R2 after training', we have successfully learnned a series of coeffecients that have produced 
    a relatively high R Squared value, meaning that the learned weights explain 7__% of the variation between the predicted
    and actual dependent variable. Congratulations, you can now use this to predict the price of your new house in Boston.
</p>

<p>
   Before I describe the processes involved in the code, I would highly recommend playing with the code above to get an 
   intuitive feeling for how it works: Try changing some of the values to understand how it impacts on the final model, 
   will more or less indepenedent variables improve or disimprove the results obtained above? You should be able to run 
   the code with the helper functions as is. 
</p>

<p>
	Now that we have the required theory out of the way, and you have (hopefully) read through the code above, we will
	walk through this code chronologically explaining all steps required to estimate our weights.
	<br>
	<bold>Note:</bold> I would recommend opening two tabs on your screen so you can see the code and the walk through
	 togetther while you read!
</p>

<p>
    <bold>Lines: 1-X</bold> These import the required libraries for this code to work.
</p>
<p>
    <bold>Lines: X-Y</bold> These import the required libraries for this code to work.
</p>