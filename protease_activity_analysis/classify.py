""" Train and test classification models """
import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn import svm, model_selection, metrics, ensemble

def classify_kfold_roc(X, Y, class_type, k_splits, pos_class):
    """Perform sample classification with k-fold cross validation and ROC analysis.

    Args:
        X: full dataset (n x m) where n is the number of samples and m is the
            number of features. Includes samples for both train/test.
        Y: true labels (n x j), where n is the number of samples and j is the
            number of classes. Includes labels for both train/test.
        class_type ("svm", "random forest"): type of classifier
        k_splits: number of splits for cross validation
        pos_class: string name for positive class

    Returns:
        probs: class probabilities for samples in test splits
        scores: prediction scores for samples in test splits
        tprs: true positive rates for each fold of cross validation, interpolated
        auc: ROC AUC for each fold of cross validation
    """
    # splits for k-fold cross validation
    cv = model_selection.StratifiedKFold(n_splits=k_splits)

    if class_type == "svm": # support vector machine
        classifier = svm.SVC(kernel='linear', probability=True,
            random_state=0)
    elif class_type == "rf": # random forest classifier
        classifier = ensemble.RandomForestClassifier(max_depth=2, random_state=0)

    probs = []
    scores = []

    # for ROC analysis
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for i, (train, test) in enumerate(cv.split(X, Y)):
        X_test = X[test]
        Y_test = Y[test]

        # fit the model and predict
        classifier.fit(X[train], Y[train])
        prob = classifier.predict_proba(X_test)
        classes = classifier.classes_.tolist()
        ind_pos = classes.index(pos_class)
        preds = prob[:,ind_pos]
        score = classifier.score(X_test, Y_test)
        probs.append(prob)
        scores.append(score)

        # ROC analysis
        fpr, tpr, _ = metrics.roc_curve(Y_test, preds, pos_label=pos_class)
        auc = metrics.auc(fpr, tpr)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc)
    return probs, scores, tprs, aucs
