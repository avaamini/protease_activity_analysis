""" Collection of data visualization and plotting functions. """
import os
import copy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import plotnine
from sklearn import preprocessing

from matplotlib.patches import Ellipse
from sklearn import svm, model_selection, metrics, ensemble
from plotnine import ggplot, geom_tile, aes, scale_fill_gradient2, coord_equal, \
    themes
from adjustText import adjust_text
import protease_activity_analysis as paa

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

### CORE PLOTTING FUNCTIONS FOR PLOTTING & VISUALIZATION ###

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
    out_path, file_name, palette=COLORS_EIGHT):
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

    return fig

def plot_volcano(data_matrix, plex, group_key, group1, group2, out_path, file_name):
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
        'Fold change (' + ' '.join(group2) + '/' + ' '.join(group1) +')')
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

    return fig

def plot_confusion_matrix(cm_df, all_classes, test_classes, out_path, file_name,
    cmap):
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
    sns.set(font_scale=2.0)
    g = sns.heatmap(cm_df, annot=True,
        xticklabels=all_classes, yticklabels=test_classes, cmap=cmap)
    g.set_yticklabels(g.get_yticklabels(), rotation = 0)
    g.set_xlabel('Predicted Label', fontsize=12)
    g.set_ylabel('True Label', fontsize=12)
    g.set_title(file_name, fontsize=14)

    file_name = file_name + "_confusion.pdf"
    fig = g.get_figure()
    fig.savefig(os.path.join(out_path, file_name))
    plt.close(fig)

    return fig

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
            label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
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

    return fig

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

def plot_heatmap(data_matrix, out_path, row_colors=None, col_colors=None, center=0, metric='euclidean',
                 method='average', scale='log2'):
    """ Plot heatmap of protease activity data, with hierarchical clustering.
    Args:
        data_matrix (pandas df): data to visualize using the heatmap
        out_path (str): path to store the results
        row_colors (list, str): colors for labeling the samples in the heatmap,
            e.g., []'g', 'royalblue', 'orange', 'orange', 'orange', 'orange',
            'g', 'g', 'g', 'g', 'g', 'orange', 'royalblue', 'orange', 'orange']
        col_colors (list, str): colors for labeling the columns in the heatmap,
            e.g., []'g', 'royalblue', 'orange', 'orange', 'orange', 'orange',
            'g', 'g', 'g', 'g', 'g', 'orange', 'royalblue', 'orange', 'orange']
        center (float): where to set the center fo the color palette
        metric (str): Distance metric to use for the data,
            See scipy.spatial.distance.pdist documentation
        method (str): Linkage method to use for calculating clusters,
            See scipy.cluster.hierarchy.linkage documentatio
        scale (str): scaling to be performed: 'None', 'log2'
    Returns:
        heat (pandas.DataFrame): data matrix for the heatmap.
    """

    # Import data
    heat = data_matrix

    # Scale data
    if scale == 'log2':
        heat = np.log2(heat)

    # Define column color labels
    if col_colors:
        col_labels = pd.DataFrame({'Substrate Class': col_colors}, index=heat.columns)

    # Define row color labels
    if row_colors:
        row_labels = pd.DataFrame({'Protease Class': row_colors}, index=heat.index)

    # Plot heatmap
    sns.set(font_scale=1.5)
    cmap = sns.diverging_palette(h_neg=260, h_pos=17, sep=5, s=99, l=55, as_cmap=True)

    if row_colors and col_colors:
        fig = sns.clustermap(heat,
                             cmap=cmap,
                             center=center,
                             linewidth=2,
                             linecolor='white',
                             dendrogram_ratio=(.15, .15),
                             figsize=(8, 8),
                             cbar_pos=(0, 0.12, .03, .4),
                             method=method,
                             row_colors=row_labels,
                             col_colors=col_labels,
                             metric=metric
                             )
    elif row_colors and not col_colors:
        fig = sns.clustermap(heat,
                             cmap=cmap,
                             center=center,
                             linewidth=2,
                             linecolor='white',
                             dendrogram_ratio=(.15, .15),
                             figsize=(8, 8),
                             cbar_pos=(0, 0.12, .03, .4),
                             method=method,
                             row_colors=row_labels,
                             metric=metric
                             )
    elif col_colors and not row_colors:
        fig = sns.clustermap(heat,
                             cmap=cmap,
                             center=center,
                             linewidth=2,
                             linecolor='white',
                             dendrogram_ratio=(.15, .15),
                             figsize=(8, 8),
                             cbar_pos=(0, 0.12, .03, .4),
                             method=method,
                             col_colors=col_labels,
                             metric=metric
                             )
    else:
        fig = sns.clustermap(heat,
                             cmap=cmap,
                             center=center,
                             linewidth=2,
                             linecolor='white',
                             dendrogram_ratio=(.15, .15),
                             figsize=(8, 8),
                             cbar_pos=(0, 0.12, .03, .4),
                             method=method,
                             metric=metric
                             )
    fig.savefig(os.path.join(out_path, 'heatmap.pdf'))

    return heat


def plot_correlation_matrix(data_matrix, title, out_path, method = 'pearson'):
    """ Plot correlation matrix of protease activity data.
    
    Args:
        data_matrix (pandas df): (normalized) cleavage data for tissue
            samples or proteases across columns and substrate across rows
        title (str): name to show on figure and saving file
        out_path (str): path to store the results
        method (str): Method of correlation to use for the data,
            See pandas.DataFrame.corr documentation
    Returns:
        corr_matrix (pandas.DataFrame): correlation coefficients for each pair
            of sample comparisons
    """
    sns.set(font_scale=1.2)
    fig = plt.figure(figsize=(12,10), dpi=200, facecolor='w', edgecolor='k')
    corr_matrix = data_matrix.corr(method = method)
    sns.heatmap(corr_matrix, annot=True)
    plt.title(str(title)+' - '+method+ ' correlation')
    plt.tight_layout()

    fig.savefig(os.path.join(out_path, title+'_corrmat_'+method+'.pdf'))

    sns.set(font_scale=1.5)

    return corr_matrix

def plot_zscore_scatter(data, out_path, corr_matrix_pearson, corr_matrix_spear):
    """ Plot scatter of z-scored protease activity data.

    Args:
        data_matrix (pandas df): (normalized) cleavage data for tissue
            samples or proteases across columns and substrate across rows
        out_path (str): path to store the results
        corr_matrix_pearson (pandas df): correlation coefficients for each pair
            of sample comparisons using Pearson's method
        corr_matrix_spear (pandas df): correlation coefficients for each pair
            of sample comparisons using Spearman's method
    """
    num_cols = len(data.columns)

    for col in range(num_cols-1):
        to_plot = np.arange(col+1,num_cols)
        for y in to_plot:
            fig, ax = plt.subplots()
            plt.scatter(data.iloc[:,col], data.iloc[:,y])
            plt.xlabel(str(data.columns[col]))
            plt.ylabel(str(data.columns[y]))
            plt.title(str(data.columns[col])+ ' vs ' + str(data.columns[y]))
            textstr = '\n'.join((r'$Pearson=%.2f$' % (corr_matrix_pearson.iloc[col,y], ),
                                 r'$Spearman=%.2f$' % (corr_matrix_spear.iloc[col,y], )))


            # change matplotlib.patch.Patch properties
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

            # place a text box in upper left in axes coords
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)
            plt.tight_layout()
            col_i = data.columns[col]
            col_y = data.columns[y]
            fig.savefig(os.path.join(out_path, f'scatter_{col_i}_vs_{col_y}.pdf')
            plt.close()
    return

def plot_zscore_hist(data_matrix, out_path, b=15, close_plot=True):
    """ Plot distribution of z-scores with histograms for each column
    (sample/protease).

    Args:
        data_matrix (pandas dataframe): (normalized) cleavage data for tissue
            samples or proteases across columns and substrate across rows
        out_path (str): path to store the results
        b (int): number of bins in plotting histogram
        close_plot (bool): if True close the resulting plot.
    """

    for col in data_matrix.columns:
        plt.figure()
        plt.hist(data_matrix[col], bins=b, histtype='bar');
        plt.xlabel('z-score')
        plt.ylabel('frequency')
        plt.title(col)
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, 'hist_'+col+'.pdf'))

    if close_plot:
        plt.close()

    return

def plot_substrate_class_pie(thr_df, dict_df, color_dict, out_path):
    """ Plots pie charts of the proportions of classes that cleaved substrates
    are in.
    
    Args:
        thr_df (pandas df): list of substrates that are significantly cleaved
            (above z-score threshold)
        dict_df (pandas df): substrate name with corresponding class label
        color_dict (dict): with keys = class and values = color for class
        out_path (str): path to store the results
    """

    # Obtain necessary dictionaries: prot_ind_map and ind_prot_map
    prot_ind_map={}
    ind_prot_map={}

    unique_classes = list(set(dict_df[dict_df.columns[0]].values))
    #     unique_classes

    for i in range(len(unique_classes)):
        ind_prot_map[i] = unique_classes[i]
        prot_ind_map[unique_classes[i]] = i

    # Map substrates to appropriate index-encoded classes
    dict_df['Index'] = dict_df[dict_df.columns[0]].map(prot_ind_map)
    converted_df = thr_df.replace(dict_df['Index'])

    for c in converted_df.columns:
        # extract given column
        one_column = converted_df[c]
        # drop any NaN values
        one_column_noNaN = one_column.dropna()
        # make all classifications consistently int values
        one_column_int = one_column_noNaN.astype(int)
        # count frequency of each classification
        counts = one_column_int.value_counts()
        l = [ind_prot_map[el] for el in counts.index]
        col = [color_dict[el] for el in l]
        # plot
        plt.figure()
        plt.pie(counts, labels=l, colors=col, autopct='%1.f%%');
        plt.title(c)
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, c+'_class_pie.pdf'))

    return


def specificity_analysis(data_matrix, out_path, threshold=1):
    """ Plots sample (protease/tissue) specificity versus cleavage efficiency.
    Args:
        data_matrix (pandas dataframe): raw cleavage data for proteases or
            tissue samples or across columns and substrate across rows
        out_path (str): path to store the results
        threshold (float): cut-off for z-scores for labeling on plot
    """

    # z-score by column (tissue sample/condition, cleavage efficency)
    cl_z = scale_data(data_matrix)

    # z-score by row (probe, substrate specificity)
    dataT = data_matrix.transpose()
    dataT = scale_data(dataT)
    sp_z = dataT.transpose()

    # number of samples
    n = data_matrix.shape[1]

    # plot for each sample
    for i in range(n):
        x = cl_z.iloc[:,i]
        y = sp_z.iloc[:,i]

        plt.figure()
        plt.scatter(x,y, s=30)
        plt.xlabel('Cleavage efficiency')
        plt.ylabel('Specificity')
        plt.title(data_matrix.columns[i])
        plt.tight_layout()

        labels = data_matrix.index
        for j, txt in enumerate(labels):
            # TO DO (optional): could change threshold to be different between x and y
            if x[j] > threshold or y[j] > threshold:
                plt.annotate(txt, (x[j], y[j]), fontsize=12)

        plt.savefig(os.path.join(out_path, 'specificity_analysis_' +
                                 str(data_matrix.columns[i]) + '.pdf'))

    return

def plot_specificity_sample(screen, out_path, threshold=1, close_plot=True, cmap=False):
    """ Plots tissue specificity versus cleavage efficiency.
    Args:
            screen (pandas df) : raw screening data
            out_path (str): path to store the results
            threshold (float): cut-off for z-scores for labeling on plot
            close_plot (bool): if True, will close the generated plot
            cmap (bool): if True, will overlay raw intensity values for the screen on scatter plot
    """
    data_matrix = screen
    raw_prot_vals = data_matrix.values
    limits = [np.min(raw_prot_vals), np.max(raw_prot_vals)]
    prot = screen.columns

    for el in prot:
        query = el

        # z-score by column (tissue sample/condition, cleavage efficency)
        cl_z = paa.vis.scale_data(data_matrix)

        # z-score by row (probe, substrate specificity)
        dataT = data_matrix.transpose()
        dataT = paa.vis.scale_data(dataT)
        sp_z = dataT.transpose()

        # get x and y coordinates for scatterplot
        x = cl_z[query]
        y = sp_z[query]

        plt.figure()

        # plot scatter plot with or without colormap
        if cmap:
            raw_query_vals = data_matrix[query].values
            plt.scatter(x, y, c=raw_query_vals, s=60, edgecolors='grey')
            plt.clim(limits[0], limits[1])
            cbar = plt.colorbar()
            cbar.set_label('Raw values in screen', fontsize=14)
        else:
            plt.scatter(x, y, s=60)

        plt.xlabel('Cleavage efficiency', fontsize=16)
        plt.ylabel('Specificity', fontsize=16)
        plt.title(query, fontsize=18)
        plt.tight_layout()

        labels = data_matrix.index
        text = []
        for j, txt in enumerate(labels):
            if x[j] > threshold or y[j] > threshold:
                text.append(plt.annotate(txt, (x[j], y[j]), fontsize=12, weight='bold'))

        adjust_text(text, force_points=4, arrowprops=dict(arrowstyle="-", color="k", lw=0.5))
        plt.savefig(os.path.join(out_path, 'specificity_analysis_' +
                                query + '.pdf'))

        if close_plot:
            plt.close()

    return

def plot_specificity_substrate(screen, out_path, threshold=1, close_plot=True, cmap=False):
    """ Plots tissue specificity versus cleavage efficiency.
        Args:
            screen (pandas df) : raw screening data
            out_path (str): path to store the results
            threshold (float): cut-off for z-scores for labeling on plot
            close_plot (bool): if True, will close the generated plot
            cmap (bool): if True, will overlay raw intensity values for the screen on scatter plot
    """
    data_matrix = screen.transpose()
    raw_prot_vals = data_matrix.values
    limits = [np.min(raw_prot_vals), np.max(raw_prot_vals)]
    subs = data_matrix.columns

    for el in subs:
        query = el

        # z-score by column (tissue sample/condition, cleavage efficiency)
        cl_z = paa.vis.scale_data(data_matrix)

        # z-score by row (probe, substrate specificity)
        dataT = data_matrix.transpose()
        dataT = paa.vis.scale_data(dataT)
        sp_z = dataT.transpose()

        # get x and y coordinates for scatterplot
        x = cl_z[query]
        y = sp_z[query]
        prot_col_map = {'Metallo': 'g', 'Serine': 'orange', 'Aspartic': 'k', 'Cysteine': 'b', 'Other': 'grey'}

        plt.figure()

        if cmap:
            raw_query_vals = data_matrix[query].values
            plt.scatter(x, y, c=raw_query_vals, s=60, edgecolors='grey')
            plt.clim(limits[0], limits[1])
            cbar = plt.colorbar()
            cbar.set_label('Raw values in screen', fontsize=14)
        else:
            plt.scatter(x, y, s=60)

        plt.xlabel('Cleavage efficiency', fontsize=16)
        plt.ylabel('Specificity', fontsize=16)
        plt.title(query, fontsize=18)
        plt.tight_layout()

        labels = data_matrix.index
        text = []
        for j, txt in enumerate(labels):
            if x[j] > threshold or y[j] > threshold:
                text.append(plt.annotate(txt, (x[j], y[j]), fontsize=12,
                            color=prot_col_map[paa.protease.classify_protease(txt)], weight='bold'))

        adjust_text(text, force_points=4, arrowprops=dict(arrowstyle="-", color="k", lw=0.5))
        plt.savefig(os.path.join(out_path, 'specificity_analysis_' +
                                 query + '.pdf'))

        if close_plot:
            plt.close()

    return

def hist(val, keys, x_label, y_label, title, identity, out_dir, close_plot=True):
    """ Specific histogram plotting function for database.py
    Args:
        val (list, array): Contains values those frequency to be plotted
        keys (list, str): list of legend labels
        x_label (str): x_label of histogram
        y_label (str): y_label of histogram
        title (str): title of histogram
        identity (str): identity of protease or susbtrate
        out_dir (str): output directory to save files to
        close_plot (bool): if True, will close the generated plot
    """
    fig, ax = plt.subplots()
    n, bins, patches = plt.hist(val, alpha=0.75)
    plt.title(title+' '+identity)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend(keys)
    plt.show()

    if out_dir:
        fig.savefig(out_dir + '/' + identity + '_histogram_zscore.pdf')

    if close_plot:
        plt.close()

    return

### HELPER FUNCTIONS FOR PLOTTING & VISUALIZATION ###

def aggregate_data(data_in_paths, out_path, name, axis=1):
    """ Combine multiple datasets into single data matrix.
    Args:
        data_in_paths (list of strings): path for datafiles
        out_path (str): path to store the results
        name (str): name for file
        axis (boolean): axes of concatenation, with True/1 as grouping
            by common substrates (horizontal) and False/0 as grouping by common
            sample names (vertical)
    Returns:
        data_matrix (pandas.DataFrame): combined data matrix
    """

    # create variables to store the compiled data/name
    frame = []

    for file_path in data_in_paths:
        # identify original file name
        file_name = os.path.basename(file_path).split('.csv')[0]
        # create pandas dataframe for each datafile
        data = pd.read_csv(file_path, index_col=0, names=['',file_name])
        # remove first row (remnants of incorrect column labels)
        data = data.iloc[1:,:]

        frame.append(data)

    # combine individual dataframes from each file into single dataframe
    agg_df = pd.concat(frame, axis=axis)

    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print('Directory created', out_path)

    # export aggregated dataframe as csv file
    data_save_path = os.path.join(out_path, f"{name}.csv")
    agg_df.to_csv(data_save_path)

    return agg_df

def process_data(data_matrix):
    """ Removes data for substrates that have negative cleavage rates across
    all samples.
    Args:
        data_matrix (pandas df): raw cleavage data for tissue samples or
            proteases across columns and substrate across rows
    Returns:
        dropping (pandas df): data matrix without substrates of consistently
            negative cleavage rates
    """
    dropping = []
    for ind in data_matrix.index:
        if all(x < 0 for x in data_matrix.loc[ind]):
            dropping.append(ind)

    return dropping

def scale_data(data_matrix):
    """ Calculates z-scores across columns (population standard deviation used).
    Args:
        data_matrix (pandas df): raw cleavage data for tissue samples or
            proteases across columns and substrate across rows
    Returns:
        scaled_data (pandas df): data matrix of normalized values (z-scores)
    """
    # Create Scaler object
    scaler = preprocessing.StandardScaler()
    # Fit data on the scaler object
    scaled_data = scaler.fit_transform(data_matrix)
    scaled_data = pd.DataFrame(scaled_data, columns=data_matrix.columns,
                               index=data_matrix.index)

    return scaled_data

def top_n_hits(data_matrix, ind_dict, out_path, n=5):
    """ Find top n cleaved substrates for each column (sample/protease).

    Args:
        data_matrix (pandas dataframe): (normalized) cleavage data for tissue
            samples or proteases across columns and substrate across rows
        ind_dict (dict): dictionary with substrate name as value and substrate
            index in data_matrix as key
        out_path (str): path to store the results
        n (int): top number of substrates to display
    Returns:
        top_hits_df (pandas df): ranked list of top n substrates by cleavage rate
    """

    top_df = pd.DataFrame()

    for c in data_matrix.columns:
        sorted_args = [i[0] for i in sorted(enumerate(data_matrix[c]),
                                            key=lambda x:x[1], reverse=True)]

        # take first n indices
        top_df[c] = sorted_args[:n]

        # convert to probe substrate name
        top_hits_df = top_df.replace(ind_dict)

        # export list as csv file
        data_save_path = os.path.join(out_path, f"top_hits.csv")
        top_hits_df.to_csv(data_save_path, index=False)

    return top_hits_df

def threshold_substrates(data_matrix, ind_dict, out_path, threshold=1):
    """ Ranks all cleaved probes above z-score threshold.

    Args:
        data_matrix (pandas dataframe): (normalized) cleavage data for tissue
            samples or proteases across columns and substrate across rows
        ind_dict (dict): dictionary with substrate name as value and substrate
            index in data_matrix as key
        out_path (str): path to store the results
        threshold (float): cut-off for cleavage z-scores
    Returns:
        thresh_df (pandas df): ranked list of substrates for all cleavage rates
            above threshold
    """
    thresh_df = pd.DataFrame()

    for c in data_matrix.columns:

        # find indices where values greater than threshold
        thr_ind = np.argwhere(np.array(data_matrix[c]) > threshold)
        thr_ind = thr_ind.flatten()
        # list values using indices
        zvals = data_matrix.iloc[list(thr_ind)][c]
        # sort indices from greatest to least
        sorted_zvals = np.argsort(-zvals)
        # find substrate names from ordered indices
        sorted_labels = sorted_zvals.index[sorted_zvals]
        thr_ind_df = pd.DataFrame(sorted_labels)
        # account for different number of substrates greater than threshold in
            # different tissues by concatenating
        thresh_df = pd.concat([thresh_df, thr_ind_df], axis=1)

    thresh_df.columns = data_matrix.columns

    # export list as csv file
    data_save_path = os.path.join(out_path, f"hits_above_threshold.csv")
    thresh_df.to_csv(data_save_path, index=False)

    return thresh_df
