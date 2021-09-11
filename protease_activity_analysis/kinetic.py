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
        """ Calculate initial rate in intensity/min
        """
        initial_rate = (self.raw_mean[self.linear_time] - self.raw_mean[0]) / (
            self.linear_time)
        initial_rate_name = 'Initial rate at t=' + str(self.linear_time)
        initial_rate_df = initial_rate.to_frame(name=initial_rate_name)
        self.inital_rate = initial_rate_df

        z_score_rate_df = self.z_score(initial_rate_df)
        self.initial_rate_zscore = z_score_rate_df

    def set_fc(self):
        """ Calculate FC metrics.
        """
        # mean fold change for all substrates at all times
        fc_mean = self.raw_mean.div(self.raw_mean[0], axis=0)
        self.fc_mean = fc_mean

        # fold change for all substrates and replicates at all times
        raw_fc = self.raw.copy()
        raw_fc = raw_fc.set_index(raw_fc.columns[0])
        fc = raw_fc.div(raw_fc[0], axis=0)
        self.fc = fc

        # fold change at time fc_time (x)
        fc_x = fc_mean[self.fc_time]
        self.fc_x = fc_x

        # Calculate z_score by fold change
        z_score_fc = self.z_score(fc_x)
        fc_name = 'Z-scored fold change at t=' + str(self.fc_time)
        z_score_fc = z_score_fc.to_frame(name=fc_name)
        self.fc_zscore = z_score_fc

        self.fc = self.fc.reset_index()

    def z_score(self, data):
        """ Standard (z) score the data

        Args:
            data (pandas.Series, pandas.Dataframe): data matrix
        Returns:
            z_s (pd series/dataframe): standardized data
        """
        z_s = ((data - data.mean()) / data.std(ddof=0))
        return z_s

    def plot_kinetic(self, kinetic_data, title, ylabel):
        """ Plot trajectory of kinetic data.

        Args:
            kinetic_data (df): data plot
            title (str): name for the plot
            ylabel (str): label for the y-axis

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

        plt.close()

        return ax

    def write_csv(self, data_to_write, save_name):
        """ Write data of interest to CSV and save """
        data_save_path = os.path.join(self.save_dir,
            f"{self.sample}_{save_name}.csv")
        data_to_write.to_csv(data_save_path)

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

    def get_inital_rate(self):
        """ Getter for initial rate """
        return self.initial_rate

    def get_intial_rate_zscore(self):
        """ Getter for initial rate, z-scored """
        return self.initial_rate_zscore


# def kinetic_visualization(data_path, col_dict, row_dict, out_dir):
#     """ Visualizes protease activity data in different formats
#
#     Args:
#         data_path: 2 options:
#             1) (str): directory of .csv file with matrix containing data to visualized where rows are substrates and
#             columns are samples. First cell contains screen name
#             2) (list, str): directories of .csv files for each sample in a given screen with a single column with name
#             of sample and rows corresponding to substrates screened. Requires building a pandas df from different inputs
#         col_dict (pandas df): labels that classify columns by some property (e.g. protease class for proteases in screen)
#         row_dict (pandas df): labels that classify rows by some property (e.g. substrate by their protease susceptibility)
#         out_dir (str): directory to save all outputs
#
#     Returns:
#
#     """
#     # TO DO: Load screening data (2 options)
#     #  aggregate_data=
#
#     # TO DO: Load name
#     # screen_name =
#
#     # TO DO: Create directory
#     save_dir = os.path.join(out_dir, screen_name)
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#         print('Directory created', save_dir)
#
#     # Generate relevant outputs and plots
#     paa.vis.plot_heatmap(aggregate_data)
#     paa.vis.plot_correlation_matrix()
#     paa.vis.plot_zscore_scatter()
#     paa.vis.plot_zscore_hist()
#     paa.vis.get_top_hits()


