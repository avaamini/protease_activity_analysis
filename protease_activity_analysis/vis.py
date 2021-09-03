""" Collection of data visualization functions """
import os
import copy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from matplotlib.patches import Ellipse
from sklearn import svm, model_selection, metrics, ensemble
from plotnine import ggplot, geom_tile, aes, scale_fill_gradient2, coord_equal, \
    themes
from adjustText import adjust_text

# Set default font to Arial
# Say, "the default sans-serif font is Arial
matplotlib.rcParams['font.sans-serif'] = "Arial"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "sans-serif"
# Sample color palette
COLORS_EIGHT = [
    'k',
    'deepskyblue',
    'lightseagreen',
    'orangered',
    'coral',
    'dodgerblue',
    'indigo',
    'darkslategray'
]


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Args:
        x, y (array-like, shape (n, )): input data
        ax (matplotlib.axes.Axes): The axes object to draw the ellipse into
        n_std (float): The number of standard deviations to determine the
            ellipse's radiuses.

    Returns:
        matplotlib.patches.Ellipse

    Other parameters:
        kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)

    return ax.add_patch(ellipse)

def plot_pca(data_matrix, features, group_key, pca_groups, biplot,
    out_path, file_name, palette=COLORS):
    """ Principal component analysis. Plot PC1 vs PC2.

    Args:
        data_matrix (pandas.pivot_table): normalized data matrix
        features (array-like, str): features to use for the PCA
        group_key (str): token key name for annotating the PCA
        pca_groups (list, str): groups to consider for the PCA
        biplot (bool): True if want to show biplot
        out_path (str): path to write the plot to
        file_name (str): token key for saving the plot
        palette: list of colors for PCA

    Returns:
        pca_scatter (fig): PCA scatter plot of data
    """

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    data_matrix = data_matrix.reset_index()
    if pca_groups is not None:
        data_matrix = data_matrix[data_matrix[group_key].isin(pca_groups)]

    x = data_matrix.loc[:,features].values
    y = data_matrix.loc[:, [group_key]].values
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=5)
    principalComponents = pca.fit_transform(x)
    finalDf = pd.DataFrame(data = {
        'principal component 1': principalComponents[:,0],
        'principal component 2': principalComponents[:,1],
        group_key: data_matrix[group_key]})

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)

    explained_variance = pca.explained_variance_ratio_
    pc1 = explained_variance[0]*100
    pc2 = explained_variance[1]*100

    ax.set_xlabel('PC1 ('+ "%0.1f" % (pc1) + '% explained var.)', fontsize = 20)
    ax.set_ylabel('PC2 ('+ "%0.1f" % (pc2) + '% explained var.)', fontsize = 20)

    ax.set_title('PCA Analysis', fontsize = 20)
    ax.tick_params(axis='both', labelsize=18)


    targets = finalDf[group_key].unique()

    colors = palette[:len(targets)]

    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['Sample Type'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
		  , c = color
                   , s = 50
                   , label = target)
        confidence_ellipse(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , ax, n_std = 2, edgecolor = color)

    if biplot:
        coeff = np.transpose(pca.components_[0:2, :])
        n = coeff.shape[0]
        for i in range(n):
            plt.arrow(0, 0, coeff[i,0]*5, coeff[i,1]*5,color = 'r',alpha = 0.5)
            plt.text(coeff[i,0]* 6, coeff[i,1] * 6, "Var"+str(i+1),
                color = 'g', ha = 'center', va = 'center')

    l = ax.legend(loc='upper right', ncol=1, handlelength=0, fontsize=14,
        frameon=False)

    for handle, text in zip(l.legendHandles, l.get_texts()):
        text.set_color(handle.get_facecolor()[0])
        text.set_ha('right')
        handle.set_color('white')

    file = file_name + "_PCA.pdf"
    fig.savefig(os.path.join(out_path, file))
    plt.close()

    return

def plot_volcano(data_matrix, group_key, group1, group2, plex, out_path, file_name):
    """ Volcano plot for differential enrichment of features between two groups.

    Args:
        data_matrix (pd.pivot_table): normalized data matrix
        group_key (str): token key name for group comparison. Found in data_matrix
        group1 (str): sample type name for the first group in the comparison
        group2 (str): sample type name for the second group in the comparison
        plex (list, str): names of the features for annotation on the volcano
        out_path (str): path to write the plot to
        file_name (str): token key for saving the plot

    Plots:
        Volcano plot of data, saved.
    """
    if group1==None and group2==None:
        targets = data_matrix.index.levels[0]
        grouped = data_matrix.groupby(level=group_key)
        group1 = targets[0]
        cond1 = grouped.get_group(targets[0])
        cond1_means = cond1.mean()

        group2 = targets[1]
        cond2 = grouped.get_group(targets[1])
        cond2_means = cond2.mean()

        fold_change = cond2_means/cond1_means
    else:
        undo_multi = data_matrix.reset_index()
        cond1 = undo_multi[undo_multi[group_key].isin(group1)]
        cond1_means = cond1.mean(axis=0)

        cond2 = undo_multi[undo_multi[group_key].isin(group2)]
        cond2_means = cond2.mean(axis=0)

        fold_change = cond2_means/cond1_means

    volcano_data = pd.DataFrame(fold_change)
    volcano_data.columns = ['Fold change']

    from scipy.stats import ttest_ind

    pvals = []

    for rep in plex :
        result = ttest_ind(cond1[rep], cond2[rep])
        pvals.append(result.pvalue)

    volcano_data.insert(1, 'P-vals', pvals)

    from statsmodels.stats.multitest import multipletests

    # Calculated the adjusted p-value using the Holm-Sidak method
    adjusted = multipletests(pvals=pvals, alpha=0.05, method="holm-sidak")

    volcano_data.insert(2, 'Adjusted p-vals', adjusted[1])
    volcano_data.insert(3, '-log10(adjP)',
        -(np.log10(volcano_data['Adjusted p-vals'])))
    volcano_data['-log10(adjP)'] = volcano_data['-log10(adjP)'].replace(np.inf, 15)

    signif = -(np.log10(0.05))

    x = volcano_data['Fold change']
    y = volcano_data['-log10(adjP)']
    sigvalues = np.ma.masked_where(y<signif, y)
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x,y,c='k',s=45)
    ax.scatter(x, sigvalues, c='r',s=45)

    texts = []
    for x, y, l in zip(x,sigvalues,volcano_data.index):
        texts.append(plt.text(x, y, l, size=12))
    adjust_text(texts)

    ax.set_xlabel(
        'Fold change (' + ' '.join(group2) + '/' + ' '.join(group1) +')',
        fontname='Arial', fontsize = 20)
    ax.set_ylabel('-log\u2081\u2080(P\u2090)', fontsize = 20)
    left,right = ax.get_xlim()
    ax.set_xlim(left=0, right = np.ceil(right))
    ax.set_title('Volcano Plot', fontsize=20)
    ax.tick_params(axis='both', labelsize=18)


    plt.rc('xtick', labelsize=40)
    plt.rc('ytick', labelsize=20)
    ax.axhline(y=1.3, linestyle='--', color='lightgray')
    ax.axvline(x=1, linestyle='--', color='lightgray')

    file = file_name + "_volcano.pdf"
    fig.savefig(os.path.join(out_path, file))
    plt.close()

    return

def plot_confusion_matrix(cm_df, all_classes, test_classes, out_path, file_name,
    cmap='Blues'):
    """ Plots confusion matrix results from multiclass classification.

    Args:
        cm_df (pandas df): dataframe containing the results of confusion matrix
            analysis for multiclass classification
        all_classes (np array): all classes in the dataset
        test_classes (np array): test classes over which the prediction occurred
        save_path (str): file path for saving the confusion matrix plot
        cmap (str): color map to use
    Plots:
        confusion matrix
    """
    ## Plot confusion matrix, average over the folds
    g = sns.heatmap(cm_df, annot=True,
        xticklabels=all_classes, yticklabels=test_classes, cmap='Blues')
    g.set_yticklabels(g.get_yticklabels(), rotation = 0)
    g.set_xlabel('Predicted Label', fontsize=12)
    g.set_ylabel('True Label', fontsize=12)
    g.set_title('Confusion Matrix Performance', fontsize=14)

    file_name = file_name + "_confusion.pdf"
    fig = g.get_figure()
    fig.savefig(os.path.join(out_path, file_name))
    fig.clf()
    plt.close(fig)

    return

def plot_kfold_roc(tprs, aucs, out_path, file_name, show_sd=True):
    """Plots mean ROC curve + standard deviation boundary from k-fold cross val.

    Args:
        tprs: true positive rates interpolated across linspace(0, 1, 100)
        aucs: ROC AUC for each of the cross validation trials
        out_path: path to directory to save plot
        file_name: file name for saving and title to show on the figure
        show_sd (bool): whether or not to show shading corresponding to sd

    """
    mean_fpr = np.linspace(0, 1, 10000)
    fig, ax = plt.subplots()

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Random chance', alpha=.8) # line for random decision boundary

    # compute average ROC curve and AUC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr) # average auc
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.3f $\pm$ %0.f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    # shading for standard deviation across the k-folds of cross validation
    if show_sd:
        std_tpr = np.std(tprs, axis=0) # already interpolated
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2,
                        label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title=file_name)
    ax.legend(loc="lower right")
    ax.set_xlabel('1 - Specificity', fontsize = 15)
    ax.set_ylabel('Sensitivity', fontsize = 15)

    file = file_name + "_ROC.pdf"
    fig.savefig(os.path.join(out_path, file))
    plt.close()

    return

def plot_matrix(data_matrix):
    """ Visualizes protease activity data matrix as heatmap.

    Args:
        data_matrix (pandas df): normalized data matrix

    Returns:
        heat_map (fig): heatmap of data
    """
    undo_multiindex = data_matrix.unstack(0)
    undo_multiindex = undo_multiindex.stack(0)
    undo_multiindex = undo_multiindex.stack(0)
    undo_multiindex = undo_multiindex.reset_index()
    undo_multiindex = undo_multiindex.rename(columns={0:"z-scores",
        "level_1": "Reporters"})
    undo_multiindex = undo_multiindex.astype({"z-scores":float})

    # by category
    fig = (
    (undo_multiindex, aes('Reporters', 'Sample Type', fill = 'z-scores'))
      + geom_tile(aes(width=0.95,  height=0.95))
      + scale_fill_gradient2(low='blue', mid = 'white', high='red', midpoint=1)
      + coord_equal()
      + theme(
              axis_ticks=element_blank(),
              axis_text_x=element_text(angle=90),
              legend_title_align='center')
    )

    fig.draw()
    plt.close()

    return fig

def plot_heatmap(data_matrix, out_path, sample_label, row_colors,
    metric='euclidean', method='average', scale='log2'):
    """ Plot heatmap of protease activity data, with hierarchical clustering.

    Args:
        data_matrix (pandas df): data to visualize using the heatmap
        out_path (str): path to store the results
        sample_label (str): type of annotation for the samples
            e.g., 'Family'
        row_colors (list, str): colors for labeling the samples in the heatmap,
            e.g., []'g', 'royalblue', 'orange', 'orange', 'orange', 'orange',
            'g', 'g', 'g', 'g', 'g', 'orange', 'royalblue', 'orange', 'orange']
        metric (str): Distance metric to use for the data,
            See scipy.cluster.hierarchy.linkage documentation
        method (str): Linkage method to use for calculating clusters,
            See scipy.spatial.distance.pdist documentatio
        scale (str): scaling to be performed: 'None', 'log2'

    Returns:
        heat (pandas.DataFrame): data matrix for the heatmap.

    """
    # TODO:
    # 1. Add color map in the matrix as the last row
    # 2. Decide on a universal color scheme + add argument for center

    # Import data
    heat = data_matrix

    # Scale data
    if scale == 'log2':
        heat = np.log2(heat)

    # Define color labels
    row_labels = pd.DataFrame({sample_label: row_colors}, index=heat.index)
    # Plot heatmap
    sns.set(font_scale=1.5)

    cmap = sns.diverging_palette(h_neg=240, h_pos=20, sep=15, s=99, l=50, as_cmap=True)
    fig = sns.clustermap(heat,
                        cmap=cmap,
                        center=0.1,
                        linewidth=2,
                        linecolor='white',
                        dendrogram_ratio=(.15, .15),
                        figsize=(8, 8),
                        cbar_pos=(0, 0.12, .03, .4),
                        method=method,
                        row_colors=row_labels,
                        metric=metric
                    )

    fig.savefig(out_path)
    plt.close()

    return heat
