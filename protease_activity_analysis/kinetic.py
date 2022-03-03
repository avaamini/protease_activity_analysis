""" Representation and analysis of kinetic protease activity data. """
import os
import copy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import protease_activity_analysis as paa

from adjustText import adjust_text

class KineticDataset:
    """ Dataset of kinetic protease activity measurements. """
    def __init__(self, data_path, fc_time, linear_time, out_dir, blank=0):
        self.data_path = data_path
        raw = pd.read_excel(data_path)
        self.raw = raw

        self.fc_time = fc_time
        self.linear_time = linear_time
        self.blank = blank

        # Screen metadata
        self.info = str(raw.columns[0]).split('-')
        self.sample_name = self.info[0] # sample screened

        self.save_dir = os.path.join(out_dir, self.sample_name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print('Directory created', self.save_dir)

        # basic metrics on the screen
        self.raw_mean = self.raw.groupby(self.raw.columns[0]).agg([np.mean])
        self.raw_mean.columns = self.raw_mean.columns.droplevel(1)

        # calculate metrics
        self.set_rate()
        self.set_fc()

    def set_rate(self):
        """ Calculate and set initial rate in fluorescence intensity (RFU)/min"""

        col = self.raw.columns

        # set initial rate according to the linear time inputted
        if self.linear_time in col:
            initial_rate = (self.raw_mean[self.linear_time] - self.raw_mean[0]) / (
                self.linear_time)
            initial_rate_name = 'Initial rate at t=' + str(self.linear_time)
            initial_rate_df = initial_rate.to_frame(name=initial_rate_name)
            self.initial_rate = initial_rate_df

            z_score_rate_df = self.z_score(initial_rate_df)
            self.initial_rate_zscore = z_score_rate_df
        else: # invalid time provided -- prompt to try again
            self.initial_rate = None
            self.initial_rate_zscore = None
            print('Initial rates set to None. Please reinitialize dataset \
                with valid time. Valid times are: {}'.format(col.to_list()))

    def set_fc(self):
        """ Calculate fold change (FC) metrics. """

        # mean fold change for all substrates at all times
        fc_mean = self.raw_mean.div(self.raw_mean[0], axis=0)
        self.fc_mean = fc_mean

        # fold change for all substrates and replicates at all times
        raw_fc = self.raw.copy()
        raw_fc = raw_fc.set_index(raw_fc.columns[0])
        fc = raw_fc.div(raw_fc[0], axis=0)
        self.fc = fc

        # fold change at time fc_time (x)
        col = self.fc_mean.columns
        if self.fc_time in col:
            fc_x = fc_mean[self.fc_time]
            self.fc_x = fc_x

            # Calculate z_score by fold change
            z_score_fc = self.z_score(fc_x)
            fc_name = 'Z-scored fold change at t=' + str(self.fc_time)
            z_score_fc = z_score_fc.to_frame(name=fc_name)
            self.fc_zscore = z_score_fc

            self.fc = self.fc.reset_index()
        else:
            print('Give valid time in screen to calculate fold change metrics')
            print('Valid times are:', col.to_list())
            self.fc_x = None
            self.fc_zscore= None
            print('Fold change metrics set to None. Please reinitialize \
                Dataset with valid time.')

    def z_score(self, data):
        """ Standard (z) score the data
        Args:
            data (pandas.Series, pandas.Dataframe): data matrix
        Returns:
            z_s (pd series/dataframe): standardized data
        """
        z_s = ((data - data.mean()) / data.std(ddof=0))
        return z_s

    def plot_kinetic(self, kinetic_data, title, ylabel, close_plot=False):
        """ Plot trajectory of kinetic data.
        Args:
            kinetic_data (df): data plot
            title (str): name for the plot
            ylabel (str): label for the y-axis
            close_plot (bool): if True, close the plot window
        Returns:
            ax (matplotlib axes): the plot
        """

        def double_std(array):
            """ Helper function 2x. std"""
            return np.std(array) * 2

        # Calculate the average and the std of replicates
        mean = kinetic_data.groupby(kinetic_data.columns[0]).agg([np.mean])
        mean.columns = mean.columns.droplevel(1)
        std = kinetic_data.groupby(kinetic_data.columns[0]).agg([double_std])
        std.columns = std.columns.droplevel(1)

        # Plot data
        mean_t = mean.T
        ax = mean_t.plot(
            legend=True,
            marker='.',
            markersize=10,
            figsize=(7, 5),
            yerr=std.T
        )

        ax.legend(loc='upper left', prop=fm.FontProperties(family='Arial'), fontsize=8)
        ax.set_xlabel('Time (min)', fontname='Arial', fontsize=14)
        ax.set_ylabel(ylabel, fontname='Arial', fontsize=14)
        ax.set_title(title, fontname='Arial', fontsize=15)
        file_path = os.path.join(self.save_dir,
            f"{title}_{ylabel}_kinetic.pdf")
        ax.figure.savefig(file_path)

        if close_plot:
            plt.close()

        return ax

    def write_csv(self, data_to_write, save_name):
        """ Write data of interest to CSV and save

        Args:
            data_to_write (df): data to write to csv
            save_name (str): name token for saving
        """
        data_save_path = os.path.join(self.save_dir,
            f"{self.sample_name}_{save_name}.csv")
        data_to_write.to_csv(data_save_path)

    def get_sample_name(self):
        """ Getter for sample name """
        return self.sample_name

    def get_raw(self):
        """ Getter for Raw data """
        return self.raw

    def get_fc(self):
        """ Getter for FC """
        return self.fc

    def get_fc_mean(self):
        """ Getter for mean FC """
        return self.fc_mean

    def get_fc_time(self):
        """ Getter for FC time """
        return self.fc_time

    def get_fc_x(self):
        """ Getter for FC at time specified """
        return self.fc_x

    def get_fc_zscore(self):
        """ Getter for z-score FC values"""
        return self.fc_zscore

    def get_initial_rate(self):
        """ Getter for initial rate """
        return self.initial_rate

    def get_initial_rate_zscore(self):
        """ Getter for initial rate, z-scored """
        return self.initial_rate_zscore

# removed col_dict for simplicity
def kinetic_visualization(data_path, screen_name, data_dir, row_dict=None,
                          n=5, b=15, threshold=1, drop_neg=True):
    """ Visualizes protease activity data in different formats. Given directory
    of .csv files for sample screens, aggregate the data and then process and
    visualize.

    Args:
        data_path (list of str): paths of .csv files for each sample
            in a kinetic screen with a single column with name of sample and rows
            corresponding to substrates screened.
        screen_name (str): name of screen. will be used for saving
        data_dir (str): directory of files

        row_dict (pandas df): labels that classify rows by some property
            (e.g. substrate by their protease susceptibility)
        n (int): number of top substrates to return in top_n_substrates
        b (int): number of bins in plotting histogram
        threshold (float): lower bound cut-off for cleavage z-scores (drop lower)
        drop_neg (boolean): if True drop substrates with a negative initial rate

    Returns:
        top_n_hits (pandas df): the top N substrates screened, ranked by zscore
        thresh_df (pandas df): screen dataset thresholded by z-score cutoff
        row_dict (pandas df): updated df of labels that classify rows by
            property of interest
        ind_dict (pandas Series): indices of the threshold and scaled kinetic
            dataset df
    """

    # Create data matrix from 1+ datafiles
    agg_df = paa.vis.aggregate_data(data_path, data_dir)

    # remove non-cleaved substrates
    if drop_neg:
        dropping = paa.vis.process_data(agg_df)
        processed_data = agg_df.drop(index=dropping)
        row_dict = row_dict.drop(index=dropping)
    else:
        processed_data = agg_df

    # z-score data
    scaled_data = paa.vis.scale_data(processed_data)
    data_save_path = os.path.join(out_dir, screen_name+"_z_scored.csv")
    scaled_data.to_csv(data_save_path, index=False)

    ind_dict = pd.Series(scaled_data.index, index=range(scaled_data.shape[0])).to_dict()

    # Aggregate data, generate relevant outputs and plots
    agg_df_t = agg_df.T

    # heatmap
    paa.vis.plot_heatmap(agg_df_t, out_dir)

    # correlation matrices
    corr_matrix_pearson = paa.vis.plot_correlation_matrix(scaled_data, screen_name, out_dir, method='pearson')
    corr_matrix_spear = paa.vis.plot_correlation_matrix(scaled_data, screen_name, out_dir, method='spearman')

    # zscore visualizations: scatter and histogram plots
    paa.vis.plot_zscore_scatter(scaled_data, out_dir, corr_matrix_pearson, corr_matrix_spear)
    paa.vis.plot_zscore_hist(scaled_data, out_dir, b)

    # top hits
    top_n_hits = paa.vis.top_n_hits(scaled_data, ind_dict, out_dir, n, plot=False)
    thresh_df = paa.vis.threshold_substrates(scaled_data, ind_dict, out_dir, threshold)

    # specificity analyses
    paa.vis.plot_specificity_sample(agg_df, out_dir, threshold=1, plot=False, cmap=True)
    paa.vis.plot_specificity_substrate(agg_df, out_dir, threshold=1, plot=False, cmap=True)

    return top_n_hits, thresh_df, row_dict, ind_dict
