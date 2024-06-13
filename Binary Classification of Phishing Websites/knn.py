import numpy as np
import pandas as pd
from collections import Counter

class KNN:
    '''A K-nearest neighbour classifier.

    Parameters
    ----------
    n_neighbors: int, default=3
        Number of neighbors between observation with shortest distance.

    
    dist_type: {euclidean, manhattan}, default=euclidean
        Choose between calculating Euclidean and Manhattan distances.
    
    References
    ----------
        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier
    '''
    def __init__(self, n_neighbors=3, dist_type='euclidean'):
        self.n_neighbors = n_neighbors
        self.dist_type = dist_type

    def fit(self, X, y):
        ''' Fit model by storing observations of X and their corresponding target y.
    
            Parameters
            ----------
            X : {array-like, sparse matrix}, shape (n_samples, n_features)
                Training data.

            y : ndarray of shape (n_samples,)
                Target values.
        '''
        self.X_train = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        self.y_train = y.to_numpy() if isinstance(y, pd.Series) else y

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
        self.X_test = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        distances = self.compute_distances()
        y_pred = []

        for distances_row in distances:
            indexes = np.argsort(distances_row)[:self.n_neighbors]
            k_nearest_labels = self.y_train[indexes]
            votes = Counter(k_nearest_labels)
            y_pred.append(votes.most_common(1)[0][0])

        return np.array(y_pred)
    
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
        self.X_test = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        distances = self.compute_distances()
        y_proba = []

        for distances_row in distances:
            indexes = np.argsort(distances_row)[:self.n_neighbors]
            k_nearest_labels = self.y_train[indexes]

            # Calculate inverse distances as weights
            weights = 1 / (distances_row[indexes] + 1e-10)  # Adding a small value to avoid division by zero
            class_probs = {}

            for label in np.unique(self.y_train):
                class_probs[label] = np.sum(weights[k_nearest_labels == label])

            # Normalize probabilities to sum to 1
            sum_weights = np.sum(weights)
            if sum_weights > 0:
                class_probs = {label: prob / sum_weights for label, prob in class_probs.items()}
            else:
                class_probs = {label: 1.0 / len(class_probs) for label in class_probs}

            y_proba.append(list(class_probs.values()))

        return np.array(y_proba)

    def compute_distances(self):
        ''' Calculate the distances between the test data and training neighbors.

            Parameters
            ----------
                This method makes use of self-stored variables self.X_test and
                self.X_train to make calculations.

            Returns
            -------
            distances : ndarray of shape (n_samples,)
                Vector containing the distances of test data between neighbors.
        '''
        if self.dist_type == 'euclidean':
            distances = np.sqrt(np.sum((self.X_test[:, np.newaxis] - self.X_train) ** 2, axis=2))
        elif self.dist_type == 'manhattan':
            distances = np.sum(np.abs(self.X_test[:, np.newaxis] - self.X_train), axis=2)
        else:
            raise ValueError('Invalid Distance Type')
        return distances