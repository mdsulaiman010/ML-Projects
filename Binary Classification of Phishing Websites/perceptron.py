import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    ''' A perceptron classifier with linear activation function.

        Parameters
        ----------
        
        weights : {'random', 'ones', 'zeros'}, default=None
            The initial weights set for the model.

        bias : int, default=None
            The initial bias introduced for the model.

        set_early_stop : bool, default=False
            Whether to use early stopping to terminate training when validation score
            is not improving. If set to True, algorithm will require validation set 
            to evaluate accuracy with updated weights at the end of an epoch.
        
        eval_set : {array-like, sparse matrix}, shape (n_samples, n_features)
            The set used for calculating validation accuracy during early stopping.

        n_early_stop : int, default=10
            Number of iterations with no improvment to wait before early stopping.

        tolerance : float, default=1e-3
            The stopping criterion. If it is not None, then iterations will stop when 
            (current_accuracy <= best_accuracy + tolerance) based on n_early_stop.

        random_state : int, default=None
            Used to shuffle the training data when ``shuffle`` is set to ``True``. 
            Pass an int for reproducible results after training the model.

        learning_rate : float, default=1.0
            Constant by which the updates are multiplied

        max_iter : int, default=1000
            The maximum number of iterations (a.k.a epochs) to train the model over 
            the training data.
        
        shuffle : bool, default=True
            Whether or not the training data should be shuffled after each epoch.

        References
        ----------
        https://en.wikipedia.org/wiki/Perceptron
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron

    '''
    def __init__(self, max_iter=1000, learning_rate=1.0, random_state=None, shuffle=True, weights=None, bias=0, set_early_stop=False, n_early_stop=10, eval_set=None, tolerance=0.001):
        self.W = weights
        self.b = bias
        self.set_early_stop = set_early_stop
        self.eval_set = eval_set
        self.n_early_stop = n_early_stop
        self.tolerance = tolerance
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.shuffle = shuffle

    def fit(self, X, y, calc_accuracy=False):
        ''' Fit model based on linear combination of bias, weights and input.
    
            Parameters
            ----------
            X : {array-like, sparse matrix}, shape (n_samples, n_features)
                Training data.

            y : ndarray of shape (n_samples,)
                Target values.
            
            calc_accuracy : bool, default=False
                Whether to calculate the training accuracy at the end of each 
                epoch.

            Returns
            -------
            W : array
                Returns the corresponding updated weights after model training.
        '''
        # Convert dataframe inputs to NumPy arrays
        self.X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        self.y = y.to_numpy() if isinstance(y, pd.Series) else y
        self.acc_list = []

        # Initializing weights
        if self.W == 'random':
            self.W = np.random.rand(self.X.shape[1])
        elif self.W == 'ones':
            self.W = np.ones(self.X.shape[1])
        else:
            self.W = np.zeros(self.X.shape[1])

        # Initialize conditions for early stopping
        if self.set_early_stop == True and self.eval_set is not None:
            eval_X, eval_y = self.eval_set
            best_accuracy = 0
            no_improvement_count = 0

        # Start of training loop based on epochs set from input
        for epoch in range(self.max_iter):
            
            if self.random_state is not None and self.random_state >= 0:
                np.random.seed(self.random_state)
            else: pass

            if self.shuffle is True:
                shuffle = np.random.permutation(len(X))
                self.X = self.X[shuffle]
                self.y = self.y[shuffle]
            else: pass

            for features, target in zip(self.X, self.y):
                y_pred = self.activation(features)
                if target != y_pred:
                    # If the prediction made is wrong, update weights and bias for predictions with next example
                    update_step = self.learning_rate * (target - y_pred)
                    self.W += update_step * features
                    self.b += update_step

            if self.set_early_stop == True:
                val_acc = self.model_accuracy(eval_y,self.predict(eval_X))
                if val_acc > best_accuracy + self.tolerance: # If there is improvement in accuracy, update best accuracy, no need to add to counter
                    best_accuracy = val_acc
                else:
                    no_improvement_count += 1
                if no_improvement_count >= self.n_early_stop and self.verbose == 1:
                    print(f'Early stopping triggered at Epoch {epoch}. Best training accuracy: {best_accuracy}')
                    break
            if calc_accuracy is True:
                self.acc_list.append(self.model_accuracy(self.y, self.predict(self.X)))
        # This updates the whole class, thus making the model retain the rebalanced weights and bias
        return str('--Training Complete--\n\n Final Weights: \n{}'.format(self.W))

    def predict(self, X):
        '''Predict class labels for samples in X.

            Parameters
            ----------
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                The data matrix used to make predictions.
            
            Returns
            -------
            y_pred : ndarray of shape (n_samples,)
                Vector containing the class labels for each sample.
        '''
        # Input X will be converted to NumPy array if it is a DataFrame. Otherwise, it will be left alone
        example_set = X.to_numpy() if isinstance(X, pd.DataFrame) else X 
        
        # For each row in the example set, run it through the activation function to obtain a prediction of y
        y_preds = [self.activation(features) for features in example_set]
        return np.array(y_preds)

    def predict_proba(self, X):
        '''Calculate probabilities of selecting class label for observations.

            Parameters
            ----------
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                The data matrix used to make predictions.
            
            Returns
            -------
            y_proba : {array-like, sparse matrix} of shape (n_samples, n_labels)
                Matrix containing the probabilies of predicting each label for all
                observations.
        '''
        X_input = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        # For each row in the example set, run it through the activation function to obtain a prediction of y
        raw_scores = [np.dot(features, self.W) + self.b for features in X_input]
        # Assuming binary classification, convert raw scores to probabilities using a sigmoid function
        probs = 1 / (1 + np.exp(-np.array(raw_scores)))
        return probs

    def model_accuracy(self, true_y, pred_y):
        '''Calculate accuracy based on actual and predicted label values.

            Parameters
            ----------
            true_y : ndarray of shape (n_samples,)
                The actual labels from each observation in the data.

            pred_y : ndarray of shape (n_samples,)
                The predicted labels based on fitted model.
            
            Returns
            -------
            accuracy : float
                The proportion of correct label predictions against total
                observations from data.
        '''
        # If input y values are of different types, standardize by converting to NumPy arrays
        self.true_y = true_y.to_numpy() if isinstance(true_y,pd.Series) else true_y
        self.pred_y = pred_y.to_numpy() if isinstance(pred_y,pd.Series) else pred_y
        # Calculate accuracy
        accuracy = (np.sum(true_y == pred_y)) / len(true_y)
        return accuracy
        
    def activation(self, X):
        '''Activation function of linear combination of bias, weights and input.

            Parameters
            ----------
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                Training data. This method is meant for internal algorithm use,
                but can be used externally as an activation function.

            Returns
            -------
            int : {-1, 1}
                Based on the activation, return 1 if the sum is greater than 0.
                Otherwise, return -1.
        '''
        return np.where((np.dot(X,self.W) + self.b) > 0, 1, -1)
    
    def diagnostics(self, label=None):
        '''Plots accuracy convergence of the model.

            Parameters
            ----------
            label : str
                Represents the label of the graph if planned to plot against
                other diagnostic plots.

            Returns
            -------
            A plot of model accuracy against epoch
        '''
        plt.plot(self.acc_list, label=label,alpha=0.6)
        plt.xlabel('Epochs'); plt.ylabel('Accuracy')
        plt.ylim([0,1])
        if self.set_early_stop is True:
            plt.xlim([0,self.n_early_stop])

    def __str__(self):
        return "--Current Perceptron Settings--\nMaximum number of iteration(s): {}\nLearning rate: {}".format(self.epochs, self.learning_rate)