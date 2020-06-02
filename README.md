# Heart disease detection through neural network 1

This project is a continuation of [Heart disease detection through logistic regression 1](https://github.com/simenjh/heart-disease-regression-1) and [Heart disease detection through logistic regression 2](https://github.com/simenjh/heart-disease-regression-2)

This is a flexible multi-layer neural network implementation from scratch, using a minimum number of libraries. 


## The purposes of this project:
* Train a more advanced model for the heart disease problem.
* Compare the performance (accuracy) of this more advanced model to the polynomial models trained by logistic regression.
* If helpful, compute the F1 score of the model.

Dataset: [Heart disease UCI](https://www.kaggle.com/ronitf/heart-disease-uci)

<br />

## Run program
Call heart_disease(data_file) in heart_disease.py.

<br /> <br />

## Results
![](images/learning_curves.png?raw=true)


* The above learning curves were obtained by training a neural network with one hidden layer. The hidden layer has 25 activation units.
* As can be seen, the model has low variance, and the variance is decreasing with increasing number of training examples.
* The accuracy on the training set is 86% - 90%, and the accuracy on the cross validation set is 80% - 87%. The fluctuations are a result of initial parameters initialization.
* The model is regularized with L2 regularization.

<br />

## Possible improvements
* Hyperparameter tuning.
* More data.

<br />

The accuracy of the trained neural network isn't particulary good for the important problem of detecting heart disease. There will be too many false positives and false negatives. As such, there really is no point in computing the F1 score to measure the balance between precision and recall.

The dataset has 303 examples, which likely isn't enough for this problem. 
