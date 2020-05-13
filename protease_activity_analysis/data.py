""" Collection of data loading and processing functions """
import numpy as np
import pandas

# for future: load function can be made more modular for other data formats
def load_syneos(data_path, id_path, sheet_names, stock_id):
    """ Read a Syneos file from a path and extract data

    Args:
        data_path (str): path to the Syneos xlsx file
        id_path (str): path to the SampleType xlsx file
        sheet_names (list, str): sheets to read
        stock_id (str): name of ABN stock identifier for normalization

    Returns:
        data_matrix (pd.DataFrame)
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

def process_data_batches(data_paths, id_paths, sheet_names, stock_ids):
    """ Process several Syneos files, extract data

    Args:
        data_paths (list, str): list of paths to Syneos xlsx files
        id_paths (list, str): list of paths to SampleType files
        sheet_names (list, str): sheets to read in each file
        stock_inds (list, str): list of strings of stock names for normalization

    Returns:
        syneos_data data frame for each results files
    """

    # TODO: this is just a loop through list of files
    raise NotImplementedError
