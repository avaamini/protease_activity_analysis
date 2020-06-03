""" Collection of data loading and processing functions """
import numpy as np
import pandas as pd
import itertools


# for future: load function can be made more modular for other data formats
def load_syneos(data_path, id_path, sheet_names, stock_id):
    """ Read a Syneos file from a path and extract data

    Args:
        data_path (str): path to the Syneos xlsx file
        id_path (str): path to the SampleType xlsx file
        sheet_names (list, str): sheets to read
        stock_id (str): name of ABN stock identifier for normalization

    Returns:
        data_matrix (pandas.pivot_table)
    """

    # read syneos excel file
    usecols = [2,3,6,7,8]

    sheet_data = pandas.read_excel(data_path,
        sheet_names, header=1, usecols=usecols)

    df = None
    for key, data in sheet_data.items():
        if df is None:
            df = data
        else:
            df = df.append(data)

    # read SampleType file
    sample_to_type = pandas.read_excel(id_path, header=0, index_col=0)
    sample_type = sample_to_type.loc[df["Sample ID"]]
    df['Sample Type'] = sample_type.values

    # account for dilution factors
    replace_inds = ~np.isnan(df['Area Ratio'])
    df.loc[replace_inds,'Ratio'] = df.loc[replace_inds,'Area Ratio']

    # create data_matrix n x m where m is the number of reporters
    data_matrix = pandas.pivot_table(df,
        values='Ratio',
        index=['Sample Type', 'Sample ID'],
        columns='Compound')
    return data_matrix

def process_syneos_data(data_matrix, features_to_use):
    """ Process syneos data. Keep relevant features and mean-normalize

    Args:
        data_matrix (pandas.df): syneos MS data w/ sample ID and type
        features_to_use (list, str): reporters to include

    Returns:
        norm_data_matrix (pandas.df)
    """
    # only include the reporters that are actually part of the panel
    new_matrix = pandas.DataFrame()

    for i in range(len(features_to_use)):
         if features_to_use[i] in data_matrix.columns :
             new_matrix[features_to_use[i]] = data_matrix[features_to_use[i]]

    # perform mean normalization
    num_reporters = len(new_matrix.columns)
    num_samples = len(new_matrix.index)
    row_means = new_matrix.mean(axis = 1)

    mean_normalized = pandas.DataFrame(index=np.arange(num_samples), columns=np.arange(num_reporters))

    for i in range(num_samples):
        for j in range(num_reporters):
            mean_normalized.iat[i,j] = new_matrix.iat[i,j]/row_means.iloc[i]
            
    mean_normalized=mean_normalized[:-1]         

    return mean_normalized

def partition_data(data_matrix, p):
    """ Partition data for training and testing classifiers

    Args:
        data_matrix (pandas.df): annotated w/ labels to partition wrt
        p (float): proportion of samples to be included in training set

    Returns:
        df_train (pandas.df): training data
        df_test (pandas.df): test data
    """

    raise NotImplementedError

def filter_data(data_matrix, classes):
    """ Filter data frame according to desired classes.

    Args:
        data_matrix (pandas.df)
        classes (list, str): classes we want to included

    Returns:
        df_filtered (pandas.df): filtered data frame
    """

    raise NotImplementedError


def read_names_from_uniprot(data_path, sheet_names):
    """ Read a excel file with Uniprot data and extract all gene names

    Args:
        data_path (str): path to the uniprot xlsx file most be exported from Uniprot and contain a "Gene name column"
        sheet_names (list, str): sheets to read

    Returns:
        all_names (dict): key (sheet), value (list of names)
    """

    # Import excel file
    sheet_data = pd.read_excel(data_path, sheet_names,
                               header=0)

    all_names = {}
    for sheet in sheet_names:
        data = sheet_data[sheet]
        # Create a new column with the nuber of names for each inhibitor
        data['n'] = data["Gene names"].apply(lambda x: x.split(" "))

        names = list(itertools.chain.from_iterable(data['n'].values))
        all_names[sheet] = names

    return all_names