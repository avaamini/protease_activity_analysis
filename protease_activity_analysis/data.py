""" Collection of data loading and processing functions """
import numpy as np
import pandas as pd
import itertools


# for future: load function can be made more modular for other data formats
def load_syneos(data_path, id_path, sheet_names):
    """ Read a Syneos file from a path and extract data

    Args:
        data_path (str): path to the Syneos xlsx file
        id_path (str): path to the SampleType xlsx file
        sheet_names (list, str): sheets to read

    Returns:
        data_matrix (pandas.pivot_table)
    """

    # read syneos excel file
    usecols = [2,3,6,7,8]

    sheet_data = pd.read_excel(data_path,
        sheet_names, header=1, usecols=usecols)

    df = None
    for key, data in sheet_data.items():
        if df is None:
            df = data
        else:
            df = df.append(data)

    # read SampleType file
    sample_to_type = pd.read_excel(id_path, header=0, index_col=0)
    sample_type = sample_to_type.loc[df["Sample ID"]]
    df['Sample Type'] = sample_type.values

    # account for dilution factors
    replace_inds = ~np.isnan(df['Area Ratio'])
    df.loc[replace_inds,'Ratio'] = df.loc[replace_inds,'Area Ratio']

    # create data_matrix n x m where m is the number of reporters
    data_matrix = pd.pivot_table(df,
        values='Ratio',
        index=['Sample Type', 'Sample ID'],
        columns='Compound')
    return data_matrix

def process_syneos_data(data_matrix, features_to_use, stock_id,
    sample_type_to_use=None, sample_ID_to_use=None):
    """ Process syneos data. Keep relevant features and mean-normalize

    Args:
        data_matrix (pandas.df): syneos MS data w/ sample ID and type
        features_to_use (list, str): reporters to include
        stock_id (str): Sample Type ID for stock to use for normalization
        sample_type_to_use (list, str): sample types to use
        sample_ID_to_use (str): contains (sub)string indicator of samples to
            include, e.g. "2B" or "2hr" to denote 2hr samples. default=None

    Returns:
        norm_data_matrix (pandas.df)
    """

    def eliminate_zero_row(row):
        """Identifies whether a row has two or more features that are zero-valued.

        Args:
            row: data frame rows of syneos data

        Returns:
            log: array of booleans indicating whether row has two or more features
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
    stock = filtered_matrix.loc[('Stock',stock_id)].to_numpy()
    filtered_matrix = filtered_matrix / stock
    filtered_matrix = filtered_matrix.drop('Stock', level='Sample Type')

    # eliminate those samples that do not meet the sample ID name criterion
    if sample_ID_to_use != None:
        undo_multi = filtered_matrix.reset_index()
        filtered_matrix = undo_multi[undo_multi['Sample ID'].str.contains(
            sample_ID_to_use)]
        filtered_matrix = filtered_matrix.set_index(['Sample Type', 'Sample ID'])

    # eliminate those samples that do not meet the sample type name criterion
    if sample_type_to_use != None:
        filtered_matrix = filtered_matrix[filtered_matrix.index.isin(
            sample_type_to_use, level=0)]

    # mean normalization
    mean_normalized = filtered_matrix.div(filtered_matrix.mean(axis=1),axis=0)

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
