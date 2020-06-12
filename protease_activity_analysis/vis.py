""" Collection of data visualization functions """
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sns
from plotnine import *
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse

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
    undo_multiindex = undo_multiindex.rename(columns={0:"z-scores", "level_1": "Reporters"})
    undo_multiindex = undo_multiindex.astype({"z-scores":float})
    
    # by category 
    fig1 = (ggplot(undo_multiindex, aes('Reporters', 'Sample Type', fill = 'z-scores'))
      + geom_tile(aes(width=0.95,  height=0.95))
      + scale_fill_gradient2(low='blue', mid = 'white', high='red', midpoint=1)
      + coord_equal()
      + theme(                                         
              axis_ticks=element_blank(),
              axis_text_x=element_text(angle=90),
              legend_title_align='center')
      
    )
    
    # by individual sample 
    fig2 = (ggplot(undo_multiindex, aes('Reporters', 'Sample ID', fill = 'z-scores'))
      + geom_tile(aes(width=0.95,  height=0.95))
      + scale_fill_gradient2(low='blue', mid = 'white', high='red', midpoint=1)
      + coord_equal()
      + theme(                                         
              axis_ticks=element_blank(),
              axis_text_x=element_text(angle=90),
              legend_title_align='center')
      + coord_flip()
      
    )
    
    fig1.draw()
    #fig2.draw()

    #to_plot = reshape.pivot("Sample ID", "Reporters", "values")
    # fig, ax1 = plt.subplots(1,2)
    # sns.heatmap(to_plot, ax=ax1)
    # sns.heatmap(to_plot, ax=ax2)
    # plt.show()

    return 

def plot_pca(data_matrix, reporters):
    """ Plots and saves principal component analysis fig
    Args:
        data_matrix (pandas.pivot_table): normalized data matrix
    Returns:
        pca: PCA decomposition of data
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
    
    return finalDf

    
def plot_volcano(data_matrix):
    """ Plots and saves volcano plot figure
    Args:
        data_matrix (pandas.pivot_table): normalized data matrix
    Returns:
        volcano (fig): volcano plot of data. saved
    """

    # TODO: volcano plot
    raise NotImplementedError

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
