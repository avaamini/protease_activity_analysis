""" Collection of Syneos data loading and processing functions """
import numpy as np
import pandas as pd
import itertools
import os
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from utils import get_output_dir

class SyneosDataset:
    """ Dataset of Syneos MS measurements. For analysis/classification. """
    def __init__(self, save_dir, save_name, file_list=None, use_cols=[2,3,6,7,8]):
        self.save_dir = save_dir
        self.save_name = save_name
        self.use_cols = use_cols

        self.raw_data_matrix = None
        self.data_matrix = None
        self.mean_scaled_matrix = None
        self.z_scored_matrix = None
        self.feature_map = None

        # Load directly from files of already processed data, set data matrix
        if file_list != None:
            self.data_matrix = self.load_syneos_data(file_list)
            self.features = self.data_matrix.columns

    def load_syneos_data(self, file_list):
        """ Load and create data matrix from a list of pickle files.

        Args:
            file_list (list, str): paths to pickle files containing data
        """
        # read pickle files in file list and load the data
        matrices = []
        for f in file_list:
            data = pd.read_pickle(f)
            matrices.append(data)
        data = pd.concat(matrices)

        return data

    def read_syneos_data(self, data_path, id_path, stock_path, sheet_names):
        """ Read a Syneos file from a path and extract data

        Args:
            data_path (str): path to the Syneos xlsx file
            id_path (str): path to the SampleType xlsx file
            stock_path (str): path to the StockNormalization xlsx file
            sheet_names (list, str): sheets to read

        Returns:
            data_matrix (pandas.pivot_table)
        """

        # read syneos excel file
        usecols = self.use_cols # HARDCODED

        sheet_data = pd.read_excel(data_path,
            sheet_names, header=1, usecols=usecols)

        df = None
        indices = ['Sample Type', 'Sample ID'] #HARDCODED
        for key, data in sheet_data.items():
            if df is None:
                df = data
            else:
                df = df.append(data)

        # read SampleType file
        sample_to_type = pd.read_excel(id_path, header=0, index_col=0)
        sample_type = sample_to_type.reindex(df["Sample ID"])
        df['Sample Type'] = sample_type.values

        # read Stock file if it is specified for the normalization
        if stock_path is not None:
            stock_info = pd.read_excel(stock_path, header=0, index_col=0)
            stock = stock_info.reindex(df["Sample ID"])
            df['Stock Type'] = stock.values
            indices.append('Stock Type')

        # account for dilution factors
        replace_inds = ~np.isnan(df['Area Ratio'])
        df.loc[replace_inds,'Ratio'] = df.loc[replace_inds,'Area Ratio']

        # create data_matrix n x m where m is the number of reporters
        data_matrix = pd.pivot_table(
            df,
            values='Ratio',
            index=indices,
            columns='Compound')

        self.raw_data_matrix = data_matrix
        return data_matrix

    def process_syneos_data(self, features_to_use, stock_id, sample_type_to_use, sample_ID_to_use, sample_ID_to_exclude):
        """ Process syneos data. Keep relevant features and samples.

        Args:
            features_to_use (list, str): reporters to include
            stock_id (list, str): Sample Type ID for stock to use for normalization
            sample_type_to_use (list, str): sample types to use
            sample_ID_to_use (str): contains (sub)string indicator of samples to
                include, e.g. "2B" or "2hr" to denote 2hr samples. default=None
            sample_ID_to_exclude (list, str): specific sample IDs to exclude

        Returns:
            filtered_data_matrix (pandas.df): processed and filtered matrix of
                syneos MS data
        """

        def eliminate_zero_row(row):
            """Identifies whether a row has two or more features that are zero.

            Args:
                row: data frame rows of syneos data

            Returns:
                log: array of booleans indicating whether row has >= two features
                    that are zero-valued (True == >= 2 zero-valued features)
            """
            num_zeros = sum([x==0 for x in row])
            log = (num_zeros >= 2)
            return log

        # only include the reporters that are actually part of the panel
        new_matrix = pd.DataFrame(self.raw_data_matrix[features_to_use])

        # eliminate those samples that have two or more 0 values for the reporters
        zero_rows = np.array(new_matrix.apply(eliminate_zero_row, axis=1))
        filtered_matrix = new_matrix[~zero_rows]

        # normalize everything to stock
        if 'Stock Type' in filtered_matrix.index.names: # multiple stocks specified
            for stock in stock_id:
                stock_values = filtered_matrix.loc['Stock'].loc[stock].to_numpy()
                # find the samples matching the current stock and normalize
                filtered_matrix.loc[(
                    filtered_matrix.index.get_level_values('Stock Type')==stock)] \
                    /= stock_values
            filtered_matrix = filtered_matrix.drop('Stock', level='Sample Type')
            filtered_matrix.reset_index(inplace=True)
            filtered_matrix = filtered_matrix.drop('Stock Type', axis=1)

        else: # one stock only, no additional file
            stock_values = filtered_matrix.loc['Stock'].loc[stock_id].to_numpy()
            filtered_matrix /= stock_values
            filtered_matrix = filtered_matrix.drop('Stock', level='Sample Type')
            filtered_matrix.reset_index(inplace=True)

        # eliminate those samples that do not meet the sample type name criterion
        if sample_type_to_use != None:
            filtered_matrix = filtered_matrix[filtered_matrix['Sample Type'].isin(sample_type_to_use)]

        # eliminate those samples that do not meet the sample ID name criterion
        if sample_ID_to_use != None:
            filtered_matrix = filtered_matrix[filtered_matrix['Sample ID'].str.contains(sample_ID_to_use)]

        # eliminate those samples that are specified for exclusion
        if sample_ID_to_exclude != None:
            filtered_matrix = filtered_matrix[~filtered_matrix['Sample ID'].isin(sample_ID_to_exclude)]

        filtered_matrix = filtered_matrix.set_index(['Sample Type', 'Sample ID'])
        filtered_matrix.to_csv(
            os.path.join(self.save_dir, f"{self.save_name}_filtered_matrix.csv")
        )

        # Rename to the desired nomenclature
        if self.feature_map != None:
            filtered_matrix.rename(self.feature_map, axis=1, inplace=True)

        # Set and return the filtered matrix
        self.data_matrix = filtered_matrix
        self.features = list(self.data_matrix.columns)
        return filtered_matrix

    def mean_scale_matrix(self):
        """ Perform per-sample mean scaling on the data matrix.

        Returns and sets:
            mean_scaled (pandas df): per-sample mean scaled data
        Writes:
            CSV file containing the mean scaled data matrix
        """

        # mean scaling
        mean_scaled = self.data_matrix.div(self.data_matrix.mean(axis=1),axis=0)
        mean_scaled.to_csv(
            os.path.join(self.save_dir, f"{self.save_name}_mean_scaled.csv")
        )

        self.mean_scaled_matrix = mean_scaled
        return mean_scaled

    def standard_scale_matrix(self, axis=0):
        """ Perform z-scoring on the data matrix according to the specified axis.
        Defaults to z-scoring such that values for individual features are zero mean
        and standard deviation of 1, i.e., feature scaling with axis=0.

        Args:
            axis (int): axis for the standardization
        Returns and sets:
            z_scored (pandas df): z_scored data matrix
        Writes:
            CSV file containing the z-scored data matrix
        """
        # z scoring across samples
        z_scored = pd.DataFrame(stats.zscore(self.data_matrix, axis=axis))
        z_scored.columns = self.data_matrix.columns
        z_scored.index = self.data_matrix.index

        z_scored.to_csv(
            os.path.join(self.save_dir, f"{self.save_name}_z_scored_{axis}.csv")
        )

        self.z_scored_matrix = z_scored
        return z_scored

    def data_to_pkl(self, save_name):
        """ Write data matrices to pickle files

        Args:
            save_name (str): token name for saving files

        Writes:
            pkl files for data matrices
        """
        if self.data_matrix is not None:
            self.data_matrix.to_pickle(
                os.path.join(self.save_dir, f"{save_name}.pkl")
            )
        if self.mean_scaled_matrix is not None:
            self.mean_scaled_matrix.to_pickle(
                os.path.join(self.save_dir, f"{save_name}_mean.pkl")
            )
        if self.z_scored_matrix is not None:
            self.z_scored_matrix.to_pickle(
                os.path.join(self.save_dir, f"{save_name}_zscore.pkl")
            )

    def data_to_csv(self, save_name):
        """ Write data matrices to csv files

        Args:
            save_name (str): token name for saving files

        Writes:
            csv files for data matrices
        """
        if self.data_matrix is not None:
            self.data_matrix.to_csv(
                os.path.join(self.save_dir, f"{save_name}.csv")
            )
        if self.mean_scaled_matrix is not None:
            self.mean_scaled_matrix.to_csv(
                os.path.join(self.save_dir, f"{save_name}_mean.csv")
            )
        if self.z_scored_matrix is not None:
            self.z_scored_matrix.to_csv(
                os.path.join(self.save_dir, f"{save_name}_zscore.csv")
            )

    def get_data_matrix(self):
        """ Get the filtered data matrix """
        return self.data_matrix

    def get_raw_matrix(self):
        """ Get the original, unfiltered, unscaled matrix. """
        return self.raw_data_matrix

    def get_mean_scaled_matrix(self):
        """ Get the mean scaled data matrix """
        return self.mean_scaled_matrix

    def get_zscored_matrix(self):
        """ Get the z scored data martix """
        return self.z_scored_matrix

    def set_feature_mapping(self, feature_map_path):
        """ Reads a csv file with N-plex reporter and name mapping.

        Args:
            feature_map_path (str): path to the file
        Returns:
            plex (list str): the reporters (i.e., 1UR3_01, etc)
            renamed (list str): renamed reporter names (i.e., PP01, etc)
        """
        reporters = pd.read_csv(feature_map_path, header=0)
        plex = reporters["Reporter"].tolist()
        renamed = reporters["Name"].tolist()

        self.features_original = plex # reporters, 1UR3_01, etc
        self.features_renamed = renamed # renamed mapping, PP01, etc
        self.feature_map = defaultdict(str)
        for original, new in zip(plex, renamed):
            self.feature_map[original] = new

        return plex, renamed

    def set_original_features(self, feature_list):
        """ Set the feature names

        Args:
            feature_list (list str): names of features
        """
        self.features_original = feature_list

    def set_renamed_features(self, feature_list):
        """ Set the feature names

        Args:
            feature_list (list str): names of features
        """
        self.features_renamed = feature_list

    def get_original_features(self):
        """ Get the original features (named in original reporter format)"""
        return self.features_original

    def get_renamed_features(self):
        """ Get the renamed features """
        return self.features_renamed

    def get_features(self):
        """ Get the features used in the dataset """
        return self.features

    def make_multiclass_dataset(self, classes_include, test_types=None):
        """ Create a dataset for multiclass classification.
        Args:
            classes_include (list, str): list of labels for dataset.
                this should include the test_classes if test_classes are desired.
            test_types (list, str): list of labels to be held out for testing,
                if any
            use_mean (bool): whether to use mean scaled data

        Returns:
            X (np array): matrix of size n (samples) x m (features) of the data
            Y (np array): matrix of size n (samples) x k (classes)
            data (pd dataframe): pandas data frame containing X, Y, class labels, and
                sample IDs
            X_test (np array): matrix of size n x m of the test data
            Y_test (np array): matrix of size n x k containing the true
                labels for the samples, where k is the number of classes
            data_test (pd dataframe): df containing X, Y, labels, ID
        """
        data = self.data_matrix

        # get Sample Type, ID and copy for class labeling
        sample_type = data.index.get_level_values('Sample Type').to_numpy()
        class_labels = np.copy(sample_type)
        sample_id = data.index.get_level_values('Sample ID').to_numpy()

        # eliminate the data that is not in the classes_include
        class_inds = [i for i, val in enumerate(class_labels) if val in classes_include]
        class_labels = class_labels[class_inds]

        # prepare the headers
        data = data.reset_index()
        data = data.iloc[class_inds, :]
        data['Class Labels'] = class_labels
        data_index_headers = ['Sample Type', 'Sample ID', 'Class Labels']
        if 'Stock Type' in data.keys():
            data_index_headers.append('Stock Type')

        # create separate data frame for the test data
        X_test = None
        Y_test = None
        data_test = None
        if test_types != None:
            data_test = data.copy()
            test_inds = []
            for i, val in enumerate(sample_id):
                has_type = [(test_type in val) for test_type in test_types]
                if any(has_type):
                    test_inds.append(i)
            mask = data_test.index.isin(test_inds)

            data_test = data_test[mask]
            data_test = data_test.set_index(data_index_headers)

            data = data[~mask]

            X_test = data_test.to_numpy()
            Y_test = data_test.index.get_level_values('Class Labels').to_numpy()

        data = data.set_index(data_index_headers)

        X = data.to_numpy()
        Y = data.index.get_level_values('Class Labels').to_numpy()

        return X, Y, data, X_test, Y_test, data_test

    def make_class_dataset(self, pos_classes=None, pos_class=None,
        neg_classes=None, neg_class=None, test_types=None):
        """ Creates a dataset for binary classification.

        Args:
            pos_classes (list, str): list of Sample Type labels for
                the positive class
            pos_class (str): what to rename the positive class
            neg_classes (list, str): list of Sample Type labels for
                the negative class
            neg_class (str): what to rename the negative class
            test_types (list, str): list of Sample Type labels for
                separate test cohort


        Returns:
            X (np array): matrix of size n (samples) x m (features) of the data
                for training/validation
            Y (np array): matrix of size n x 1 containing the class labels
                for the samples in the dataset, for binary classification problem.
                for training/validation
            data (pd dataframe): pandas data frame containing X, Y, Sample Type, and
                Sample Type ID
            X_test (np array): matrix of size n x m of the data. for test.
                Defaults to None if test_classes=None
            Y_test (np array): matrix of size n x 1 containing the class labels
                labels for the samles in the dataset, for binary classification
                problems. for testing.
            data_test (pd dataframe): pandas data frame containing X_test, Y_test,
                Sample Type, and Sample Type ID
        """
        data = self.data_matrix

        # get Sample Type, ID, and copy for class labeling
        sample_type = data.index.get_level_values('Sample Type').to_numpy()
        sample_id = data.index.get_level_values('Sample ID').to_numpy()
        class_labels = np.copy(sample_type)

        # convert positive classes if necessary
        if pos_classes != None:
            pos_inds = [i for i, val in enumerate(sample_type) if val in pos_classes]
            class_labels[pos_inds] = pos_class
        # convert negative classes if necessary
        if neg_classes != None:
            neg_inds = [i for i, val in enumerate(sample_type) if val in neg_classes]
            class_labels[neg_inds] = neg_class

        # eliminate the data that is not in the neg_class, the pos_class
        classes = [neg_class, pos_class]
        class_inds = [i for i, val in enumerate(class_labels) if val in classes]
        class_labels = class_labels[class_inds]

        # prepare the data and return
        data = data.reset_index()
        data = data.iloc[class_inds, :]
        data['Class Labels'] = class_labels
        data_index_headers = ['Sample Type', 'Sample ID', 'Class Labels']
        if 'Stock Type' in data.keys():
            data_index_headers.append('Stock Type')

        # separate into the train/val dataset and the test dataset (consisting of
        #   specified sample types
        X_test = None
        Y_test = None
        data_test = None

        if test_types != None:
            data_test = data.copy()
            test_inds = []
            for i, val in enumerate(sample_id):
                has_type = [(test_type in val) for test_type in test_types]
                if any(has_type):
                    test_inds.append(i)

            mask = data_test.index.isin(test_inds)

            data_test = data[mask]
            data_test = data_test.set_index(data_index_headers)

            data = data[~mask]

            X_test = data_test.to_numpy()
            Y_test = data_test.index.get_level_values('Class Labels').to_numpy()

        data = data.set_index(data_index_headers)
        X = data.to_numpy()
        Y = data.index.get_level_values('Class Labels').to_numpy()

        return X, Y, data, X_test, Y_test, data_test
