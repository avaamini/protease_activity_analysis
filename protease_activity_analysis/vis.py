""" Collection of data visualization functions """
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sns
from plotnine import *


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

    # TODO: PCA
    
    undo_multiindex = data_matrix.reset_index()
    
    from sklearn.preprocessing import StandardScaler
    
    features = reporters
    x = undo_multiindex.loc[:,features].values 
    y = undo_multiindex.loc[:, ['Sample Type']].values 
    x = StandardScaler().fit_transform(x)
    
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pandas.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
    finalDf = pandas.concat([principalDf, undo_multiindex[['Sample Type']]], axis = 1)
    
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = finalDf['Sample Type'].unique()
    colors = ['r', 'g', 'b', 'y']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['Sample Type'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c = color
                   , s = 50)
    ax.legend(targets)
    ax.grid()

    
    print(undo_multiindex.head)
    
    return x, finalDf

    
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
