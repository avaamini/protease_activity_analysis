""" Collection of data visualization functions """
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sns
#from plotnine import *
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse

import os
import pandas as pd
import matplotlib.pyplot as plt

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
    
    from plotnine import ggplot, geom_tile, aes, scale_fill_gradient2, coord_equal, themes
    undo_multiindex = data_matrix.unstack(0)
    undo_multiindex = undo_multiindex.stack(0)
    undo_multiindex = undo_multiindex.stack(0)
    undo_multiindex = undo_multiindex.reset_index()
    undo_multiindex = undo_multiindex.rename(columns={0:"z-scores", "level_1": "Reporters"})
    undo_multiindex = undo_multiindex.astype({"z-scores":float})

    # by category
    fig1 = (
(undo_multiindex, aes('Reporters', 'Sample Type', fill = 'z-scores'))
      + geom_tile(aes(width=0.95,  height=0.95))
      + scale_fill_gradient2(low='blue', mid = 'white', high='red', midpoint=1)
      + coord_equal()
      + theme(
              axis_ticks=element_blank(),
              axis_text_x=element_text(angle=90),
              legend_title_align='center')

    )

    # by individual sample
   # fig2 = (ggplot(undo_multiindex, aes('Reporters', 'Sample ID', fill = 'z-scores'))
    #  + geom_tile(aes(width=0.95,  height=0.95))
    #  + scale_fill_gradient2(low='blue', mid = 'white', high='red', midpoint=1)
    #  + coord_equal()
   #   + theme(
  #            axis_ticks=element_blank(),
   #           axis_text_x=element_text(angle=90),
     #         legend_title_align='center')
    #  + coord_flip()

   # )

    fig1.draw()
    #fig2.draw()

    #to_plot = reshape.pivot("Sample ID", "Reporters", "values")
    # fig, ax1 = plt.subplots(1,2)
    # sns.heatmap(to_plot, ax=ax1)
    # sns.heatmap(to_plot, ax=ax2)
    # plt.show()

    return fig1

def plot_pca(data_matrix, reporters, data_path):
    """ Plots and saves principal component analysis fig
    Args:
        data_matrix (pandas.pivot_table): normalized data matrix
    Returns:
        pca_scatter (fig): PCA scatter plot of data
    """

    from sklearn.preprocessing import StandardScaler

    features = reporters
    x = data_matrix.loc[:,features].values
    y = data_matrix.loc[:, ['Sample Type']].values
    x = StandardScaler().fit_transform(x)

    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pandas.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
    finalDf = pandas.concat([principalDf, data_matrix[['Sample Type']]], axis = 1)


    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    explained_variance = pca.explained_variance_ratio_
    pc1 = explained_variance[0]*100
    pc2 = explained_variance[1]*100
    ax.set_xlabel('PC1 ('+ "%0.1f" % (pc1) + '% explained var.)', fontsize = 15)
    ax.set_ylabel('PC2 ('+ "%0.1f" % (pc2) + '% explained var.)', fontsize = 15)
    ax.set_title('PCA Analysis of Inventiv data', fontsize = 20)
    targets = finalDf['Sample Type'].unique()
    colors = ['k', 'lightseagreen', 'deepskyblue', 'steelblue', 'darkblue']
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

    l = ax.legend(loc='upper right', ncol=1, handlelength=0, fontsize=16, frameon=False)

    for handle, text in zip(l.legendHandles, l.get_texts()):
        text.set_color(handle.get_facecolor()[0])
        text.set_ha('right')
        handle.set_color('white')
    
    fig.savefig(os.path.join(data_path, "urine_pca.pdf"))
    return  


def plot_volcano(data_matrix, group1, group2, plex, data_path):
    """ Plots and saves volcano plot figure
    Args:
        data_matrix (pandas.pivot_table): normalized data matrix
    Returns:
        volcano (fig): volcano plot of data. saved
    """

    # transposed_norm = data_matrix.transpose()

    grouped = data_matrix.groupby(level="Sample Type")

    cond1 = grouped.get_group(group1)
    cond1_means = cond1.mean()

    cond2 = grouped.get_group(group2)
    cond2_means = cond2.mean()

    fold_diff = cond2_means/cond1_means
    volcano_data = pandas.DataFrame(fold_diff)
    volcano_data.columns = ['Fold difference']


    from scipy.stats import ttest_ind

    pvals = []

    for rep in plex :
        result = ttest_ind(cond1[rep], cond2[rep])
        pvals.append(result.pvalue)

    volcano_data.insert(1, 'P-vals', pvals)

    from statsmodels.stats.multitest import multipletests

    # Calculated the adjusted p-value using the Holm-Sidak method (as was done in Prism)
    adjusted = multipletests(pvals=pvals, alpha=0.05, method="holm-sidak")

    volcano_data.insert(2, 'Adjusted p-vals', adjusted[1])
    volcano_data.insert(3, '-log10(adjP)', -(np.log10(volcano_data['Adjusted p-vals'])))
    volcano_data['-log10(adjP)'] = volcano_data['-log10(adjP)'].replace(np.inf, 15)

    signif = 1.3

    x = volcano_data['Fold difference']
    y = volcano_data['-log10(adjP)']
    sigvalues = np.ma.masked_where(y<signif, y)
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x,y,c='k')
    ax.scatter(x, sigvalues, c='r')

    # for x,y,rep in zip(x,sigvalues,volcano_data.index):
    #     label = rep
    #     # this method is called for each point
    #     plt.annotate(label, # this is the text
    #                  (x,y), # this is the point to label
    #                  textcoords="offset points", # how to position the text
    #                  xytext=(0, np.random.randint(5,10)), # distance from text to points (x,y)
    #                  ha='center') # horizontal alignment can be left, right or center

    from adjustText import adjust_text

    texts = []
    for x, y, l in zip(x,sigvalues,volcano_data.index):
        texts.append(plt.text(x, y, l, size=12))
    adjust_text(texts)

    ax.set_xlabel('Fold change (' + group2 + '/' + group1 +')', fontsize = 15)
    ax.set_ylabel('-log\u2081\u2080(P\u2090)', fontsize = 15)
    left,right = ax.get_xlim()
    ax.set_xlim(left=0, right = np.ceil(right))
    ax.axhline(y=1.3, linestyle='--', color='lightgray')
    ax.axvline(x=1, linestyle='--', color='lightgray')
    
    fig.savefig(os.path.join(data_path, "volcano.pdf"))
    return 

def plot_ROC(data_matrix, reporters):
    """ Plots and saves principal component analysis fig
    Args:
        data_matrix (pandas.pivot_table): normalized data matrix
    Returns:
        pca_scatter (fig): PCA scatter plot of data
    """


def roc_curves(y_true, y_score, pos_label=None):
    """ Performs ROC analysis and plots curves

    Args:
        y_true (list, int/str): true labels. if labels are not (0,1),
            then pos_label should be explicitly given.
        y_score (list, float): class assignment scores
        pos_label (str): string for positive class

    Returns:
        roc_metrics: ROC
        roc_curves (fig): ROC curves of the data. saved
    """

    # TODO: ROC curves

    raise NotImplementedError


def render_html_figures(figs):
    """ Render a list of figures into an html document and render it.

    Args:
        figs (list, plt.Figure): a list of matplotlib figures to render
    """

    html = ""
    for fig in figs:
        html += mpld3.fig_to_html(fig)
        html += "<hr>"
    mpld3._server.serve(html)


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

    return fc, fc_x, z_score_fc, init_rate, z_score_rate
