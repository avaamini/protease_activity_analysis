""" Collection of data visualization functions """
import os
import copy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.transforms as transforms
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse
from sklearn import svm, model_selection, metrics, ensemble
from plotnine import ggplot, geom_tile, aes, scale_fill_gradient2, coord_equal, \
    themes
from adjustText import adjust_text

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
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

def plot_heatmap(data_matrix, reporters):
    """ Plots and saves heat map fig

    Args:
        data_matrix: normalized data matrix

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

def plot_pca(data_matrix, reporters, pca_groups, data_path, file_name):
    """ Plots and saves principal component analysis fig
    Args:
        data_matrix (pandas.pivot_table): normalized data matrix
    Returns:
        pca_scatter (fig): PCA scatter plot of data
    """

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    features = reporters
    data_matrix = data_matrix.reset_index()
    if pca_groups is not None:
        data_matrix = data_matrix[data_matrix['Sample Type'].isin(pca_groups)]

    x = data_matrix.loc[:,features].values
    y = data_matrix.loc[:, ['Sample Type']].values
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    finalDf = pd.DataFrame(data = {
        'principal component 1': principalComponents[:,0],
        'principal component 2': principalComponents[:,1],
        'Sample Type': data_matrix['Sample Type']})

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    explained_variance = pca.explained_variance_ratio_
    pc1 = explained_variance[0]*100
    pc2 = explained_variance[1]*100
    ax.set_xlabel('PC1 ('+ "%0.1f" % (pc1) + '% explained var.)', fontsize = 20)
    ax.set_ylabel('PC2 ('+ "%0.1f" % (pc2) + '% explained var.)', fontsize = 20)
    ax.set_title('PCA Analysis of Inventiv data', fontsize = 20)
    ax.tick_params(axis='both', labelsize=18)

    targets = finalDf['Sample Type'].unique()
    # COLORS accounts for 10 groups; @Melodi can revise as you wish
    #COLORS = ['k', 'lightseagreen', 'deepskyblue', 'steelblue', 'darkblue',
    #    'dodgerblue', 'indigo', 'lightsalmon', 'lime', 'darkslategray']
    #COLORS = ['darkblue','k','orange']
    COLORS = ['k', 'deepskyblue', 'lightseagreen', 'orangered', 'coral', 'dodgerblue', 'indigo', 'darkslategray']
    colors = COLORS[:len(targets)]

    print(finalDf)
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

    l = ax.legend(loc='upper right', ncol=1, handlelength=0, fontsize=16,
        frameon=False)

    for handle, text in zip(l.legendHandles, l.get_texts()):
        text.set_color(handle.get_facecolor()[0])
        text.set_ha('right')
        handle.set_color('white')

    file = file_name + "_PCA.pdf"
    fig.savefig(os.path.join(data_path, file))
    plt.close()

    return

def plot_volcano(data_matrix, group1, group2, plex, data_path, file_name):
    """ Plots and saves volcano plot figure
    Args:
        data_matrix (pd.pivot_table): normalized data matrix
    Returns:
        volcano (fig): volcano plot of data. saved
    """
    if group1==None and group2==None:
        targets = data_matrix.index.levels[0]
        grouped = data_matrix.groupby(level="Sample Type")
        group1 = targets[0]
        cond1 = grouped.get_group(targets[0])
        cond1_means = cond1.mean()

        group2 = targets[1]
        cond2 = grouped.get_group(targets[1])
        cond2_means = cond2.mean()

        fold_change = cond2_means/cond1_means
    else:
        undo_multi = data_matrix.reset_index()
        cond1 = undo_multi[undo_multi['Sample Type'].isin(group1)]
        cond1_means = cond1.mean(axis=0)
        print(cond1_means)

        cond2 = undo_multi[undo_multi['Sample Type'].isin(group2)]
        print(undo_multi)
        cond2_means = cond2.mean(axis=0)
        print(cond2_means)

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

    ax.set_xlabel('Fold change (' + ' '.join(group2) + '/' + ' '.join(group1) +')', fontsize = 20)
    ax.set_ylabel('-log\u2081\u2080(P\u2090)', fontsize = 20)
    left,right = ax.get_xlim()
    ax.set_xlim(left=0, right = np.ceil(right))
    ax.tick_params(axis='both', labelsize=18)
    # Hardcoded for AIP experiment
    # ax.set_ylim(bottom=-0.2831143480741629, top=np.ceil(5.990397227318687))
    

    # plt.rc('xtick', labelsize=40)
    # plt.rc('ytick', labelsize=20)
    ax.axhline(y=1.3, linestyle='--', color='lightgray')
    ax.axvline(x=1, linestyle='--', color='lightgray')

    print(volcano_data)
    file = file_name + "_volcano.pdf"
    fig.savefig(os.path.join(data_path, file))
    plt.close()

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
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    # shading for standard deviation across the k-folds of cross validation
    if show_sd:
        std_tpr = np.std(tprs, axis=0) # already interpolated
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
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

def kinetic_analysis(in_path, out_path, fc_time, linear_time, blank=0):
    """ Analyze kinetic data based on fold change + initial rate

        Args:
            - in_path (str): path to raw data, columns = timepoints, rows = substrates, cell 0,0 = name of "protease/sample - probe name" e.g. 'MMP13-Substrate'
            - out_path (str): path to store the results
            - fc_time (int): time in min at which to take the fold change
            - linear_time (int): time in min to take initial speed
            - blank (int): 1 = blank data provided, 0 = no blank data provided [default]


        Returns:
            - fc (pandas.DataFrame): Fold change for all samples in fluorescence from time 0
            - fc_x (pandas.DataFrame): Fold change at time x
            - z_score_fc (pandas.core.series.Series): Z_score of fold change at fc_time
            - init_rate (pandas.core.series.Series): Initial rate
            - z_score_rate (pandas.core.series.Series): Z_score of initial rates
        """
    def plot_kinetic(data, title, ylabel, path):
        # Calculate the average and the std of replicates
        mean = data.groupby(data.columns[0]).agg([np.mean])
        mean.columns = mean.columns.droplevel(1)
        std = data.groupby(data.columns[0]).agg([double_std])
        std.columns = std.columns.droplevel(1)

        # Plot data
        mean_t = mean.T
        ax = mean_t.plot(legend=True, marker='.', markersize=10, figsize=(7, 5), yerr=std.T)
        ax.legend(loc='best', fontsize=12)
        ax.set_xlabel('Time (min)', fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(title, fontsize=15)
        ax.figure.savefig(path)

        plt.close()

        return ax

    # Calculate z score of pandas.Series or pandas.Dataframe
    def z_score(data):
        z_s = ((data - data.mean()) / data.std(ddof=0))
        return z_s

    def double_std(array):
        return np.std(array) * 2

    # Import kinetic data
    raw = pd.read_excel(in_path)

    # Get name of sample being screened against and probe-type being screened
    info = str(raw.columns[0]).split('-')
    prot = info[0]
    # print('prot:', prot)
    sub = info[1]
    # print('sub:', sub)

    # Create new directory to save outputs with name of 'prot'
    path = os.path.join(out_path, prot)
    if not os.path.exists(path):
        os.makedirs(path)
        print('Directory created', path)

    # Plot raw kinetic data
    plot_kinetic(raw, prot, 'Intensity', path=str(path) + '/' + str(prot) + '_raw_kinetic_data.pdf')

    # Find initial rate in intensity/min
    raw_mean = raw.groupby(raw.columns[0]).agg([np.mean])
    raw_mean.columns = raw_mean.columns.droplevel(1)
    init_rate = (raw_mean[linear_time] - raw_mean[0]) / (linear_time)
    init_rate = init_rate.to_frame(name='Initial rate')

    # Calculate z_score based on init_rate
    z_score_rate = z_score(init_rate)
    # z_score_rate= z_score_rate.to_frame(name='Z-scored initial rate')

    # Find the mean fold change for all substrates at all times
    fc_mean = raw_mean.div(raw_mean[0], axis=0)

    # Find the fold change for all substrates and replicates at all times
    raw2 = raw.set_index(raw.columns[0])
    fc = raw2.div(raw2[0], axis=0)

    # Find the fold-change at time fc_time (x)
    fc_x = fc_mean[fc_time]

    # Calculate z_score by fold change
    z_score_fc = z_score(fc_x)
    z_score_fc = z_score_fc.to_frame(name='Z-scored fold change')

    # Plot fc kinetic data
    data = fc.reset_index()
    plot_kinetic(data, prot, 'Fold change', path=str(path) + '/' + str(prot) + '_fc_kinetic_data.pdf')

    fc.to_csv(str(path) + '/' + str(prot) + '_fc.csv')
    fc_x.to_csv(str(path) + '/' + str(prot) + '_fc_x.csv')
    z_score_fc.to_csv(str(path) + '/' + str(prot) + '_z_score_fc.csv')
    init_rate.to_csv(str(path) + '/' + str(prot) + '_init_rate.csv')
    z_score_rate.to_csv(str(path) + '/' + str(prot) + '_z_score_rate.csv')

    plt.close()

    return fc, fc_x, z_score_fc, init_rate, z_score_rate

def plot_heatmap(in_path, out_path, metric='euclidean', method='average', scale='log2'):
    """ Plot heatmap of data

            Args:
                - in_path (str): path to raw data
                - out_path (str): path to store the results
                - metric (str): Distance metric to use for the data, See scipy.cluster.hierarchy.linkage documentation
                - method (str): Linkage method to use for calculating clusters, See scipy.spatial.distance.pdist documentatio
                - scale (str): scaling to be performed: 'None', 'log2'


            Returns:
                - scaled_data (pandas.DataFrame): scaled data

            TODO:
            1. Add color map in the matrix as the last row
            2. Decide on a universal color scheme + add argument for center

    """

    # Import data
    heat = pd.read_excel(in_path, index_col=0)

    # Scale data
    if scale == 'log2':
        heat = np.log2(heat)

    # Define color labels
    row_colors = pd.DataFrame({'Family': ['g', 'royalblue', 'orange', 'orange', 'orange', 'orange', 'g', 'g', 'g', 'g',
                                          'g', 'orange', 'royalblue', 'orange', 'orange']}, index=heat.index)
    # Plot heatmap
    sns.set(font_scale=1.5)

    cmap = sns.diverging_palette(h_neg=240, h_pos=20, sep=15, s=99, l=50, as_cmap=True)
    fig = sns.clustermap(heat, cmap=cmap, center=0.1, linewidth=2, linecolor='white', dendrogram_ratio=(.15, .15),
                         figsize=(8, 8), cbar_pos=(0, 0.12, .03, .4), method=method, row_colors=row_colors,
                         metric=metric)

    fig.savefig(out_path)
    plt.close()

    return heat

def render_html_figures(figs):
    """ Render a list of figures into an html document and render it.

    Args:
        figs (list, plt.Figure): a list of matplotlib figures to # REVIEW: nder
    """
    raise NotImplementedError
    # html = ""
    # for fig in figs:
    #     html += mpld3.fig_to_html(fig)
    #     html += "<hr>"
    # mpld3._server.serve(html)
