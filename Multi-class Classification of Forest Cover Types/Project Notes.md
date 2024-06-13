# This was my final year project (FYP) focused on using the covtype dataset from UCI Machine Learning Repository. #

The objective was to predict from 7 different forest cover types, for trees from the Roosevelt National Forest, with
high performance.

The challenge of using this dataset was the severe label imbalance, high dimensionality, and long training times due to 
high number of observations. The chosen metric was F1-macro scoring to properly account for observing performance while 
using the dataset's natural label imbalance.

The final result was that XGBoostClassifier performed the best among other ensemble methods and multinomial logistic regression, with F1-macro score of 0.879. For curiosity, accuracy_score was found to be 0.923, suggesting that more observations were from the majority class being correctly classified.

A point of further work would be to try using a multilayer perceptron algorithm and compare its performance with the finetuned XGBoostClassifier model. The main reason for not utilizing MLP was due to long training time and approaching deadline.
