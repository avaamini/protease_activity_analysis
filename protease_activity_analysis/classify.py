""" Train and test classification models """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm, model_selection, metrics, ensemble, linear_model, feature_selection

def multiclass_classify(X, Y, class_type, k_splits, save_path):
    """Perform multiclass sample classification with k-fold cross validation.
    Plots confusion matrix heatmap.

    Args:
        X: full dataset (n x m) where n is the number of samples and m is the
            number of features. Includes samples for both train/test.
        Y: true labels (n x j), where n is the number of samples and j is the
            number of classes. Includes labels for both train/test.
        class_type ("svm", "rf", "lr"): type of classifier
        k_splits: number of splits for cross validation
        save_path: path to save

    Returns:
        probs: class probabilities for samples in test splits
        scores: prediction scores for samples in test splits
    """
    # splits for k-fold cross validation
    cv = model_selection.StratifiedKFold(n_splits=k_splits)

    if class_type == "svm": # support vector machine
        classifier = svm.SVC(kernel='linear', probability=True,
            random_state=0)
    elif class_type == "rf": # random forest classifier
        classifier = ensemble.RandomForestClassifier(max_depth=2, random_state=0)
    elif class_type == "lr": # logistic regression with L2 loss
        classifier = linear_model.LogisticRegression(random_state=0)

    probs = []
    scores = []
    cms = []

    for i, (train, test) in enumerate(cv.split(X, Y)):
        X_test = X[test]
        Y_test = Y[test]

        # fit the model and predict
        classifier.fit(X[train], Y[train])
        prob = classifier.predict_proba(X_test) # prediction probabilities
        score = classifier.score(X_test, Y_test) # accuracy
        pred = classifier.predict(X_test) # predicted class
        cm = metrics.confusion_matrix(Y_test, pred) # confusion matrix
        cm_norm = cm / cm.sum(axis=1, keepdims=1) # normalized

        probs.append(prob)
        scores.append(score)
        cms.append(cm_norm)

    cms = np.asarray(cms)
    cms_avg = np.mean(cms, axis=0)
    cm_df = pd.DataFrame(data = cms_avg)
    classes = np.unique(Y)

    ## Plot confusion matrix, average over the folds
    g = sns.heatmap(cm_df, annot=True, xticklabels=classes, yticklabels=classes, cmap='Blues')
    g.set_yticklabels(g.get_yticklabels(), rotation = 0)
    g.set_xlabel('Predicted Label', fontsize=12)
    g.set_ylabel('True Label', fontsize=12)

    file = save_path + "_confusion.pdf"
    fig = g.get_figure()
    fig.savefig(file)

    return probs, scores, cms

def classify_kfold_roc(X, Y, class_type, k_splits, pos_class):
    """Perform sample binary classification with k-fold cross validation and ROC analysis.

    Args:
        X: full dataset (n x m) where n is the number of samples and m is the
            number of features. Includes samples for both train/test.
        Y: true labels (n x j), where n is the number of samples and j is the
            number of classes. Includes labels for both train/test.
        class_type ("svm", "rf", "lr"): type of classifier
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
    elif class_type == "lr": # logistic regression with L2 loss
        classifier = linear_model.LogisticRegression(random_state=0)

    probs = []
    scores = []

    # for ROC analysis
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 10000)

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

def recursive_feature_elimination(X, Y, class_type, k_splits, out_path, save_path):
    """Recursive feature elimination. Tunes the number of features selected
            using k-fold cross validation.

    Args:
        X: full dataset (n x m) where n is the number of samples and m is the
            number of features. Includes samples for both train/test.
        Y: true labels (n x j), where n is the number of samples and j is the
            number of classes. Includes labels for both train/test.
        class_type ("svm", "rf", "lr"): type of classifier
        k_splits: number of splits for cross validation
        save_path: path to save file

    Returns:

    """
    # splits for k-fold cross validation
    cross_val = model_selection.StratifiedKFold(n_splits=k_splits)

    if class_type == "svm": # support vector machine
        classifier = svm.SVC(kernel='linear', probability=True)
    elif class_type == "rf": # random forest classifier
        classifier = ensemble.RandomForestClassifier(max_depth=2)
    elif class_type == "lr": # logistic regression with L2 loss
        classifier = linear_model.LogisticRegression()

    # use accuracy, which is reflective of number of correct classifications
    rfe_cv = feature_selection.RFECV(estimator=classifier, step=1, \
        cv=cross_val, scoring='accuracy')

    rfe_cv.fit(X, Y)
    print("Optimal number of features : %d" % rfe_cv.n_features_)
    import pdb; pdb.set_trace()

    ## Plot # of reporters vs. accuracy
    g = sns.lineplot(x=range(1, len(rfe_cv.grid_scores_)+1), y=rfe_cv.grid_scores_)
    g.set_xlabel('Number of reporters', fontsize=12)
    g.set_ylabel('Cross validation accuracy', fontsize=12)
    g.set_xticks(range(1, len(rfe_cv.grid_scores_)+1))
    g.set_title('Recursive feature elimination')

    file = save_path + "_rfe.pdf"
    fig = g.get_figure()
    fig.savefig(os.path.join(out_path, file))
    plt.show()
    plt.close()

    return
