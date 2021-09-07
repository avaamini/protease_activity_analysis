""" Collection of data loading and processing functions """
import numpy as np
import pandas as pd
import itertools
import os
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler

from utils import get_output_dir

def load_syneos(data_path, id_path, stock_path, sheet_names):
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
    usecols = [2,3,6,7,8] # HARDCODED

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

    return data_matrix

def process_syneos_data(data_matrix, features_to_use, stock_id, out_dir,
    sample_type_to_use=None, sample_ID_to_use=None, sample_ID_to_exclude=None, save_name=None):
    """ Process syneos data. Keep relevant features and mean-normalize

    Args:
        data_matrix (pandas.df): syneos MS data w/ sample ID and type
        features_to_use (list, str): reporters to include
        stock_id (list, str): Sample Type ID for stock to use for normalization
        sample_type_to_use (list, str): sample types to use
        sample_ID_to_use (str): contains (sub)string indicator of samples to
            include, e.g. "2B" or "2hr" to denote 2hr samples. default=None
        sample_ID_to_exclude (list, str): specific sample IDs to exclude
        save_name (str): string token for saving files

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
    new_matrix = pd.DataFrame(data_matrix[features_to_use])

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
        filtered_matrix = filtered_matrix[filtered_matrix['Sample Type'].isin(
            sample_type_to_use)
        ]

    # eliminate those samples that do not meet the sample ID name criterion
    if sample_ID_to_use != None:
        filtered_matrix = filtered_matrix[filtered_matrix['Sample ID'].str.contains(
            sample_ID_to_use)]

    # eliminate those samples that are specified for exclusion
    if sample_ID_to_exclude != None:
        filtered_matrix = filtered_matrix[~filtered_matrix['Sample ID'].isin(
            sample_ID_to_exclude)
        ]

    filtered_matrix = filtered_matrix.set_index(['Sample Type', 'Sample ID'])
    filtered_matrix.to_csv(os.path.join(out_dir, f"{save_name}_filtered_matrix.csv"))

    return filtered_matrix

def mean_scale_matrix(data_matrix, out_dir, save_name):
    """ Perform per-sample mean scaling on a data matrix.

    Args:
        data_matrix (pandas df): processed syneos data matrix
        out_dir (str): directory for saving outputs
        save_name (str): string token for saving file
    Returns:
        mean_scaled (pandas df): per-sample mean scaled data
    Writes:
        CSV file containing the mean scaled data matrix
    """

    # mean scaling
    import pdb; pdb.set_trace()

    mean_scaled = data_matrix.div(data_matrix.mean(axis=1),axis=0)
    mean_scaled.to_csv(os.path.join(out_dir, f"{save_name}_mean_scaled.csv"))

    return mean_scaled

def standard_scale_matrix(data_matrix, out_dir, save_name, axis=0):
    """ Perform z-scoring on a data matrix according to the specified axis.
    Defaults to z-scoring such that values for individual features are zero mean
    and standard deviation of 1, i.e., feature scaling with axis=0.

    Args:
        data_matrix (pandas df): processed syneos data matrix
        save_name (str): string token for saving file
        axis (int): axis for the standardization
    Returns:
        z_scored (pandas df): z_scored data matrix
    Writes:
        CSV file containing the z-scored data matrix
    """
    # z scoring across samples
    z_scored = pd.DataFrame(stats.zscore(data_matrix, axis=axis))
    z_scored.columns = data_matrix.columns
    z_scored.index = data_matrix.index

    z_scored.to_csv(os.path.join(out_dir, f"{save_name}_z_scored_{axis}.csv"))

    return z_scored

def get_scaler(batch):
    """ Standardize features by removing the mean and scaling to unit variance.
    Provide StandardScaler that can then be applied to other data.

    Args:
        batch (np.array): batch of data to generate the scaler with respect to,
            N x M where N: number of samples; M: number of features
    Returns:
        scaler (StandardScaler): standard scaler (0 mean, 1 var) on the batch
    """
    scaler = StandardScaler()
    scaler.fit((batch))
    return scaler

def make_multiclass_dataset(data_dir, file_list, classes_to_include,
    test_types=None):
    """ Creates a dataset for multiclass classification.

    Args:
        data_dir (str): path to the directory where pickle files are
        file_list (list, str): list of pickle file names as in data_dir
        classes_to_include (list, str): list of Sample Type labels to consider.
            this should include the test_classes if test_classes are desired.
        test_types (list, str): list of Sample Type labels to be held out for
            testing. if None, samples with those Sample Types will be included
            in the train/validation

    Returns:
        X (np array): matrix of size n x m of the data, where n is the number of
            samples and m is the number of features
        Y (np array): matrix of size n x k containing the classification labels
            for the samples in the dataset, where k is the number of classes
        data (pd dataframe): pandas data frame containing X, Y, Sample Type, and
            Sample Type ID
        X_test (np array): matrix of size n x m of the test data
        Y_test (np array): matrix of size n x k containing the true classification
            labels for the samples, where k is the number of classes
        data_test (pd dataframe): df containing X, Y, Sample Type, Sample Type ID
    """
    # read pickle files in file list and load the data
    matrices = []
    for f in file_list:
        path = os.path.join(data_dir, f)
        data = pd.read_pickle(path)
        matrices.append(data)
    data = pd.concat(matrices)

    # get Sample Type, ID and copy for class labeling
    sample_type = data.index.get_level_values('Sample Type').to_numpy()
    class_labels = np.copy(sample_type)
    sample_id = data.index.get_level_values('Sample ID').to_numpy()

    # eliminate the data that is not in the classes_to_include
    class_inds = [i for i, val in enumerate(class_labels) if val in classes_to_include]
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

def make_class_dataset(data_dir, file_list, pos_classes=None, pos_class=None,
    neg_classes=None, neg_class=None, test_types=None):
    """ Creates a dataset for binary classification.

    Args:
        data_dir (str): path to the directory where pickle files are
        file_list (list, str): list of pickle file names as in data_dir
        pos_classes (list, str): list of Sample Type labels to be considered in
            the positive class
        pos_class (str): what to rename the positive class
        neg_classes (list, str): list of Sample Type labels to be considered in
            the negative class
        neg_class (str): what to rename the negative class
        test_types (list, str): list of Sample Type labels to be used for the
            separate test cohort

    Returns:
        X (np array): matrix of size n x m of the data, where n is the number of
            samples and m is the number of features. for training/validation
        Y (np array): matrix of size n x 1 containing the classification labels
            for the samples in the dataset, for binary classification problem.
            for training/validation
        data (pd dataframe): pandas data frame containing X, Y, Sample Type, and
            Sample Type ID
        X_test (np array): matrix of size n x m of the data, where n is the number
            of samples and m is the number of features. for test. Defaults to
            None if test_classes=None
        Y_test (np array): matrix of size n x 1 containing the classification
            labels for the samles in the dataset, for binary classification
            problems. for testing.
        data_test (pd dataframe): pandas data frame containing X_test, Y_test,
            Sample Type, and Sample Type ID
    """
    # read pickle files in file list and load the data
    matrices = []
    for f in file_list:
        path = os.path.join(data_dir, f)
        data = pd.read_pickle(path)
        matrices.append(data)
    data = pd.concat(matrices)

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

def get_plex(data_path):
    """ Reads a csv file with N-plex reporter and name mapping.

    Args:
        data_path (str): path to the file
    Returns:
        plex (list str): the reporters (i.e., 1UR3_01, etc)
        renamed (list str): renamed reporter names (i.e., PP01, etc)
    """
    reporters = pd.read_csv(data_path, header=0)
    plex = reporters["Reporter"].tolist()
    renamed = reporters["Name"].tolist()

    return plex, renamed
