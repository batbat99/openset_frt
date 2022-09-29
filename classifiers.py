from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import EVM, scipy

#Things to work on:
#add a method for both classifiers that adds a new class to the classifier
#implement a version of KDEClassifier based on KDE of statsmodels

class KDEClassifier(BaseEstimator, ClassifierMixin):
    """Bayesian generative classification based on KDE
    
    Parameters
    ----------
    bandwidth : float
        the kernel bandwidth within each class
    kernel : str
        the kernel name, passed to KernelDensity
    n : int
        the max number of training samples per class
    novelty : float
        a ratio that determine if a new sample is classified or recognized as a new class
    """
    def __init__(self, bandwidth=1.0, kernel='gaussian', n=None, shuffle=False, novelty=0):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.n = n
        self.shuffle = shuffle
        self.novelty = novelty
        
    def fit(self, X, y):
        if "tracks" in X.columns:
            X = X.drop(["tracks"], axis=1)
        X = X[y != 2]
        y = y[y != 2]        
        self.classes_ = np.sort(np.unique(y))
        training_sets = [X[y == yi] for yi in self.classes_]
        #if the novelty threshold is set then a class is added to represent all the novel elements detected
        if self.novelty:
            self.classes_ = np.hstack((self.classes_, np.array([2])))
        #if shuffle is true the training data is randomly shuffled for each class
        if self.shuffle:
            training_sets = [training_set.sample(frac=1).reset_index(drop=True) for training_set in training_sets]
        #if n is set it selects the first n elements for each class
        if self.n:
            training_sets = [training_set[:self.n] for training_set in training_sets]
        #a kerneldinsity classifier is fit for each class
        self.models_ = [KernelDensity(bandwidth=self.bandwidth,
                                      kernel=self.kernel).fit(Xi)
                        for Xi in training_sets]
        self.logpriors_ = [np.log(Xi.shape[0] / X.shape[0])
                           for Xi in training_sets]
        return self
        
    def predict_proba(self, X):
        logprobs = np.array([model.score_samples(X)
                             for model in self.models_]).T
        result = np.exp(logprobs + self.logpriors_)
        return result
        
    def predict(self, X):
        track_set = []
        track_indices = []
        if 'tracks' in X.columns:
            tracks = X["tracks"]
            tracks.reset_index(inplace=True, drop=True)
            X = X.drop(["tracks"], axis=1)
            track_set = tracks.unique()
        
        proba = self.predict_proba(X)
        
        for track in track_set:
            track_indices = tracks.index[tracks == track]
            proba[track_indices] = proba[track_indices].mean(axis=0)
        
        if self.novelty:
            proba = np.hstack((proba, np.ones((proba.shape[0], 1), dtype=proba.dtype) * self.novelty))
        return self.classes_[np.argmax(proba, 1)]
    
    

class MEVM(BaseEstimator, ClassifierMixin, EVM.MultipleEVM):
    """MEVM classifier that inherits from the MultipleEVM class for implementing extra features
    """
    def __init__(self, tailsize=40, cover_threshold=None, distance_multiplier=0.5, n=None, shuffle=False, novelty=0):
        super().__init__(tailsize=tailsize, cover_threshold = cover_threshold, distance_multiplier =distance_multiplier, distance_function=scipy.spatial.distance.euclidean)
        self.n = n
        self.shuffle = shuffle
        self.novelty = novelty
    
    def fit(self, X, y, parallel=None):
        if "tracks" in X.columns:
            X = X.drop(["tracks"], axis=1)
        X = X[y != 2]
        y = y[y != 2] 
        self.classes_ = np.sort(np.unique(y))
        training_sets = [X[y == yi] for yi in self.classes_]
        #if the novelty threshold is set then a class is added to represent all the novel elements detected
        if self.novelty:
            self.classes_ = np.hstack((self.classes_, np.array([2])))
        #if shuffle is true the training data is randomly shuffled for each class
        if self.shuffle:
            training_sets = [training_set.sample(frac=1).reset_index(drop=True) for training_set in training_sets]
        #if n is set it selects the first n elements for each class
        if self.n:
            training_sets = [training_set[:self.n] for training_set in training_sets]
        training_sets = [training_set.to_numpy() for training_set in training_sets]
        self.train(training_sets, parallel)
        return self
        
    def predict_proba(self, X):
        X = X.to_numpy()
        proba=[]
        result = self.probabilities(X)
        for p in range(len(result)):
            # compute maximum indices for all evs per evm
            max_per_ev = [np.argmax(result[p][e]) for e in range(self.size)]
            proba.append([result[p][e][max_per_ev[e]] for e in range(self.size)])
        return proba
    
    def predict(self, X):
        track_set = []
        track_indices = []
        if 'tracks' in X.columns:
            tracks = X["tracks"]
            tracks.reset_index(inplace=True, drop=True)
            X = X.drop(["tracks"], axis=1)
            track_set = tracks.unique()
        
        proba = np.array(self.predict_proba(X))
        
        for track in track_set:
            track_indices = tracks.index[tracks == track]
            proba[track_indices] = proba[track_indices].mean(axis=0)
        
        if self.novelty:
            proba = np.hstack((proba, np.ones((proba.shape[0], 1), dtype=proba.dtype) * self.novelty))
        return self.classes_[np.argmax(proba, 1)]
    
    
