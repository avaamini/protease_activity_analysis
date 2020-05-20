""" Collection of data visualization functions """
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas

def plot_heatmap(data_matrix):
    """ Plots and saves heat map fig

    Args:
        data_matrix: normalized data matrix

    Returns:
        heat_map (fig): heatmap of data
    """
    
    to_plot = data_matrix.drop(columns=["Sample ID", "Sample Type"])
    
    to_plot = to_plot.astype(float)
    
    plt.imshow(to_plot)

    return to_plot

def plot_pca(data_matrix):
    """ Plots and saves principal component analysis fig
    Args:
        data_matrix (pandas.pivot_table): normalized data matrix
    Returns:
        pca: PCA decomposition of data
        pca_scatter (fig): PCA scatter plot of data
    """

    to_plot = data_matrix.drop(columns=["Sample ID", "Sample Type"])
    
    # TODO: PCA
    raise NotImplementedError

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
