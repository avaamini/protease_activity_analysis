""" Train and test classification models """
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn import svm, model_selection, metrics, ensemble, linear_model, \
    feature_selection, pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, \
    train_test_split
from sklearn.preprocessing import StandardScaler

def get_scaler(batch):
    """ Standardize features by removing the mean and scaling to unit variance.
    Provide StandardScaler that can then be applied to other data.

    Args:
        batch (np.array): batch of data to generate the scaler with respect to,
            N x M where N: number of samples; M: number of features
    Returns:
        scaler (StandardScaler): standard scaler (0 mean, 1 var) on the batch
    """
    scaler = StandardScaler()
    scaler.fit((batch))
    return scaler

def multiclass_classify(X, Y, model_type, kernel, k_splits, save_path,
    standard_scale=False, seed=None, X_test=None, Y_test=None):
    """Perform multiclass sample classification with k-fold cross validation.

    Args:
        X: full dataset (n x m) where n is the number of samples and m is the
            number of features. Includes samples for both train/val.
        Y: true labels (n x j), where n is the number of samples and j is the
            number of classes. Includes labels for both train/val.
        model_type ("svm", "rf", "lr"): type of classifier
        kernel ("linear", "poly", "rbf"): type of kernel for svm
        k_splits: number of splits for cross validation
        save_path: path to save
        standard_scale (bool): whether or not to apply feature scaling
        seed (int): random seed for setting the random state of the classifer
        X_test: independent test dataset (n x m). Includes samples that are
            excluded from training and only for testing.
        Y_test: true labels (n x j). Includes labels for samples/classes that
            are excluded from training and only used for testing.

    Returns: val_dict, cm_df_val, test_dict, cm_df_test
        val_dict: dictionary of performance metrics for classifier evaluated on
            validation set
        cm_df_val (pandas df): confusion matrix statistics for validation set
        test_dict: dictionary of performance metrics for classifier evaluated on
            test set
        cm_df_test (pandas df): confusion matrix statistics for test set

    """
    # splits for k-fold cross validation
    cv = StratifiedKFold(n_splits=k_splits)

    probs_val = []
    scores_val = []
    cms_val = []

    probs_test = []
    scores_test = []
    cms_test = []

    classes = np.unique(Y)

    # for i, (train, val) in enumerate(cv.split(X, Y)):
    for i in range(k_splits):

        # Trials are completely independent wrt initialization of classifier.
        if model_type == "svm": # support vector machine with kernel
            classifier = svm.SVC(kernel=kernel, probability=True,
                random_state=seed)
        elif model_type == "rf": # random forest classifier
            classifier = ensemble.RandomForestClassifier(
                max_depth=2, random_state=seed)
        elif model_type == "lr": # logistic regression with L2 loss
            classifier = linear_model.LogisticRegression(random_state=seed)

        # X_train = X[train]
        # Y_train = Y[train]
        # X_val = X[val]
        # Y_val = Y[val]

        X_train, X_val, Y_train, Y_val = train_test_split(
            X, Y, test_size=0.2, shuffle=True, stratify=Y
        )

        # apply scaling
        if standard_scale:
            scaler = get_scaler(X_train)
            X_train = scaler.transform(X_train)
            X_val = scaler.transform(X_val)

        # fit the model and predict
        classifier.fit(X_train, Y_train)
        prob = classifier.predict_proba(X_val) # prediction probabilities
        score = classifier.score(X_val, Y_val) # accuracy
        pred = classifier.predict(X_val) # predicted class
        cm = metrics.confusion_matrix(Y_val, pred) # confusion matrix
        cm_norm = cm / cm.sum(axis=1, keepdims=1) # normalized

        probs_val.append(prob)
        scores_val.append(score)
        cms_val.append(cm_norm)

        # if independent test set provided
        if (X_test is not None) and (Y_test is not None):

            if standard_scale:
                X_test = scaler.transform(X_test)

            # predict on test set and return ROC metrics
            prob_test = classifier.predict_proba(X_test)
            score_test = classifier.score(X_test, Y_test)
            preds_test = classifier.predict(X_test)

            classes_test = np.unique(Y_test)
            cm_test = metrics.confusion_matrix(Y_test, preds_test)

            # classes_found = np.isin(classes, classes_test)
            # cm_test = cm_test[classes_found, :]

            cm_test_norm = cm_test / cm_test.sum(axis=1, keepdims=1) # normalized
            probs_test.append(prob_test)
            scores_test.append(score_test)
            cms_test.append(cm_test_norm)

    cms_val = np.asarray(cms_val)
    cms_avg_val = np.mean(cms_val, axis=0)
    cm_df_val = pd.DataFrame(data = cms_avg_val)

    if (X_test is not None) and (Y_test is not None):
        cms_test = np.asarray(cms_test)
        cms_avg_test = np.mean(cms_test, axis=0)
        cm_df_test = pd.DataFrame(data = cms_avg_test)
        test_classes = np.unique(Y_test)

    ## TODO: could possibly change this to a data frame
    val_dict = {"probs": probs_val, "scores": scores_val, "cms": cms_val}

    if (X_test is None) and (Y_test is None):
        test_dict = None
        cm_df_test = None
    else:
        test_dict = {"probs": probs_test, "scores": scores_test, "cms": cms_test}

    return val_dict, cm_df_val, test_dict, cm_df_test

def classify_kfold_roc(X, Y, model_type, kernel, k_splits, pos_class,
    standard_scale=False, seed=None, X_test=None, Y_test=None):
    """Binary classification with k-fold cross validation and ROC analysis.

    Args:
        X: full dataset (n x m) where n is the number of samples and m is the
            number of features. Includes samples for both train/val.
        Y: true labels (n x j), where n is the number of samples and j is the
            number of classes. Includes labels for both train/val.
        model_type ("svm", "rf", "lr"): type of classifier
        kernel ("linear", "poly", "rbf"): type of kernel for svm
        k_splits: number of splits for cross validation
        pos_class: string name for positive class
        standard_scale (bool): whether or not to apply feature scaling
        seed (int): random seed for setting the random state of the classifer
        X_test: independent test dataset (n x m). Includes samples that are
            excluded from training and only for testing.
        Y_test: true labels (n x j). Includes labels for samples/classes that
            are excluded from training and only used for testing.

    Returns:
        val_dict: dictionary of performance metrics for classifier evaluated on
            validation set (probs, scores, tprs, auc)
        test_dict: dictionary of performance metrics for classifier shown to be
            evaluated on test set (probs, scores, tprs, aucs)

            probs: class probabilities
            scores: prediction scores
            tprs: true positive rates for each fold of cross validation
            auc: ROC AUC for each fold of cross validation
    """
    # splits for k-fold cross validation
    cv = StratifiedKFold(n_splits=k_splits)

    # prediction outputs
    probs_val = []
    scores_val = []
    probs_test = []
    scores_test = []

    # for ROC analysis
    tprs_val = []
    aucs_val = []
    tprs_test = []
    aucs_test = []

    # Training and evaluation
#    for i, (train, val) in enumerate(cv.split(X, Y)):
    for i in range(k_splits):

        # Trials are completely independent wrt initialization of classifier.
        if model_type == "svm": # support vector machine with kernel
            classifier = svm.SVC(kernel=kernel, probability=True,
                random_state=seed)
        elif model_type == "rf": # random forest classifier
            classifier = ensemble.RandomForestClassifier(
                max_depth=2, random_state=seed)
        elif model_type == "lr": # logistic regression with L2 loss
            classifier = linear_model.LogisticRegression(random_state=seed)

        # X_train = X[train]
        # Y_train = Y[train]
        # X_val = X[val]
        # Y_val = Y[val]
        X_train, X_val, Y_train, Y_val = train_test_split(
            X, Y, test_size=0.2, shuffle=True, stratify=Y
        )

        # feature scaling for standardization. compute scaler on training set.
        if standard_scale:
            scaler = get_scaler(X_train)
            X_train = scaler.transform(X_train)
            X_val = scaler.transform(X_val)

        # fit the model
        classifier.fit(X_train, Y_train)
        classes = classifier.classes_.tolist()
        ind_pos = classes.index(pos_class)

        # predict on validation set and return ROC metrics
        prob_val, score_val, preds_val, interp_tpr_val, auc_val = \
            classify(classifier, X_val, Y_val, pos_class)
        probs_val.append(prob_val)
        scores_val.append(score_val)
        tprs_val.append(interp_tpr_val)
        aucs_val.append(auc_val)

        # if independent test set provided
        if (X_test is not None) and (Y_test is not None):

            if standard_scale:
                X_test = scaler.transform(X_test)

            # predict on test set and return ROC metrics
            prob_test, score_test, preds_test, interp_tpr_test, auc_test = \
                classify(classifier, X_test, Y_test, pos_class)

            probs_test.append(prob_test)
            scores_test.append(score_test)
            tprs_test.append(interp_tpr_test)
            aucs_test.append(auc_test)

    ## TODO: could possibly change this to a data frame
    val_dict = {"probs": probs_val, "scores": scores_val,
        "tprs": tprs_val, "aucs": aucs_val}

    if (X_test is None) and (Y_test is None):
        test_dict = None
    else:
        test_dict = {"probs": probs_test, "scores": scores_test,
            "tprs": tprs_test, "aucs": aucs_test}

    return val_dict, test_dict

def classify(classifier, X, Y, pos_class):
    """Use a trained classifier to make predictions on a dataset. ROC metrics.

    Args:
        X: dataset (n x m) where n is the number of samples and m is the
            number of features.
        Y: true labels (n x j), where n is the number of samples and j is the
            number of classes.
    """
    prob = classifier.predict_proba(X)
    score = classifier.score(X, Y)
    ind_pos = classifier.classes_.tolist().index(pos_class)
    preds = prob[:,ind_pos]

    # ROC metrics
    mean_fpr = np.linspace(0, 1, 10000)
    fpr, tpr, _ = metrics.roc_curve(Y, preds, pos_label=pos_class)
    auc = metrics.auc(fpr, tpr)
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0

    return prob, score, preds, interp_tpr, auc

def rfe_cv(X, Y, class_type, k_splits, out_path, save_name, standard_scale=False):
    """ Recursive feature elimination. Evaluate the accuracy of the model with
    select number of features and plot # of features vs. accuracy.

    Args:
        X: full dataset (n x m) where n is the number of samples and m is the
            number of features. Includes samples for both train/test.
        Y: true labels (n x j), where n is the number of samples and j is the
            number of classes. Includes labels for both train/test.
        class_type ("svm", "rf", "lr"): type of classifier. for SVM, uses linear
            kernel
        k_splits: number of splits for cross validation
        out_path: path to save file
        save_name (str): string token for file saving
        standard_scale (bool): whether to employ feature standardization

    """

    if class_type == "svm": # support vector machine with kernel
        classifier = svm.SVC(kernel='linear', probability=True)
    elif class_type == "rf": # random forest classifier
        classifier = ensemble.RandomForestClassifier(max_depth=2)
    elif class_type == "lr": # logistic regression with L2 loss
        classifier = linear_model.LogisticRegression()

    def get_models(base_model, max_features):
        """ Get models with varying numbers of features used.

        Args:
            base_model (sklearn model): base model type (e.g., SVM)
            max_features (int): maximum number of features allowed in the RFE
        Returns:
            models (dict): mapping # features used in model -> model RFE pipeline
        """
        models = dict()
        for i in range(2, max_features+1):
            rfe = feature_selection.RFE(estimator=base_model, n_features_to_select=i)
            models[str(i)] = pipeline.Pipeline(steps=[('s',rfe),('m',base_model)])
        return models

    # evaluate a classifier using cross-validation
    def classify_cv(model, X, Y):
        """ Evaluate classifier using cross-validation.

        Args:
            model (sklearn model): classifier to evaluate
            X (np array): n x m, num samples x num features
            Y (np array): n x 1, num samples and their class labels
        Returns:
            accuracies (nd array): accuracies for each run of the cross-validation
            aucs (nd array): ROC AUCs for each run of the cross-validation
        """
        cv = model_selection.RepeatedStratifiedKFold(n_splits=k_splits, n_repeats=3)
        accuracies = cross_val_score(
            model, X, Y, scoring='accuracy', cv=cv, error_score='raise'
        )
        aucs = cross_val_score(
            model, X, Y, scoring='roc_auc', cv=cv, error_score='raise'
        )
        return accuracies, aucs
    
    # evaluate the models and store results
    res_accuracies, res_aucs, names = list(), list(), list()
    num_features = X.shape[1]
    classifiers = get_models(classifier, num_features)

    for name, model in classifiers.items():
        accuracies, aucs = classify_cv(model, X, Y)
        res_accuracies.append(accuracies)
        res_aucs.append(aucs)
        names.append(name)

        print('>%s Accuracy: %.3f (%.3f)' % (
            name, np.mean(res_accuracies), np.std(res_accuracies)))
        print('>%s AUC: %.3f (%.3f)' % (
            name, np.mean(res_aucs), np.std(res_aucs)))

    # Plot num features vs. accracy/auc
    fig, ax = plt.subplots()
    ax.boxplot(res_accuracies, labels=names)
    ax.set_xlabel('Number of Features', fontsize = 15)
    ax.set_ylabel('Accuracy', fontsize = 15)
    fig = ax.get_figure()
    file = save_name + "_RFE_accuracy.pdf"
    fig.savefig(os.path.join(out_path, file))
    plt.close()

    fig, ax = plt.subplots()
    ax.boxplot(res_aucs, labels=names)
    ax.set_xlabel('Number of Features', fontsize = 15)
    ax.set_ylabel('ROC AUC', fontsize = 15)
    fig = ax.get_figure()
    file = save_name + "_RFE_auc.pdf"
    fig.savefig(os.path.join(out_path, file))
    plt.close()

    return
