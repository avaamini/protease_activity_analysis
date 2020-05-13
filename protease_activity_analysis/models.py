""" Train and test classification models """
import numpy as np
import pandas

def train(model, df_train):
    """ Train model on data. Models can be found/taken from scikit-learn

    Args:
        model (sklearn object): classification model to train
        df_train (pandas.df): data for training

    Returns:
        trained_model (sklearn object): trained model
    """

    raise NotImplementedError

def test(model, df_test):
    """ Evaluate trained model on held-out test set.

    Args:
        model (sklearn object): trained classifier
        df_test (pandas.df): data for testing

    Returns:
        y_pred: predicted labels
        y_score: probabilities of belonging to negative/positive class
    """

    raise NotImplementedError
