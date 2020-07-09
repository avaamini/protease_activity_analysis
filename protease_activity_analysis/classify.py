""" Train and test classification models """
import numpy as np
import pandas
from sklearn import svm, model_selection, metrics, ensemble

def classify_kfold_roc(X, Y, class_type, k_splits):
    """Perform sample classification with k-fold cross validation and ROC analysis.

    Args:
        X: full dataset (n x m) where n is the number of samples and m is the
            number of features. Includes samples for both train/test.
        Y: true labels (n x j), where n is the number of samples and j is the
            number of classes. Includes labels for both train/test.
        class_type ("svm", "random forest"): type of classifier
        k_splits: number of splits for cross validation

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
            random_state=random_state)
    elif class_type == "random forest": # random forest classifier
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
        prob = classifier.predict_proba(X_test, Y_test)
        preds = prob[:,1]
        score = classifier.score(X_test, Y_test)
        probs.append(prob)
        scores.append(score)

        # ROC analysis
        fpr, tpr, _ = metrics.roc_curve(Y_test, preds)
        auc = metrics.auc(fpr, tpr)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc)

    return probs, scores, tprs, aucs

def plot_kfold_roc(tprs, aucs, plot_title, show_sd=True):
    """Plots mean ROC curve + standard deviation boundary from k-fold cross val.

    Args:
        tprs: true positive rates interpolated across linspace(0, 1, 100)
        aucs: ROC AUC for each of the cross validation trials
        plot_title: title to show on the figure
        show_sd: whether or not to show shading corresponding to sd across trials

    Returns:
    """
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Random chance', alpha=.8) # line for random decision boundary

    # compute average ROC curve and AUC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr) # average auc
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    # shading for standard deviation across the k-folds of cross validation
    if show_sd:
        std_tpr = np.std(tprs, axis=0) # already interpolated
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title=plot_title)
    ax.legend(loc="lower right")
    plt.show()
    return fig
