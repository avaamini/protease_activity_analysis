""" Collection of data searching/lookup functions based on existing screening data"""
import pandas as pd

# for future: consider adapting to being able to take a list of  proteases or to not take a path
def search_protease(prot, screen_names, path):
    """ Look up available screening data for a protease of interest

    Args:
        prot (str): protease of interest
        screen_names (list, str): screens to parse for information
        path (str): path to data folder

    Returns:
        relevant_data (dict): dictionary with keys being screens with relevant data and values being pandas.core.series with relevant screening data
    """
    screen_data = {}
    proteases = {}
    count = 0

    # Read screening files and extract protease lists for each
    for element in screen_names:
        name = 'screen_' + str(count)
        screen_data[name] = pd.read_csv(path + 'screening_data/%s' % element)
        screen_data[name] = screen_data[name].set_index('Substrate')
        proteases[name] = screen_data[name].columns.tolist()
        count += 1

    # Extract information relevant to protease and store in relevant_data
    relevant_data = {}
    for element in screen_data:
        if prot in proteases[element]:
            relevant_data[element] = screen_data[element][prot]

    return relevant_data


# for future: consider adapting to being able to take a list of  substrates or to not take a path
def search_substrate(subs, screen_names, path):
    """ Look up available screening data for a substrate of interest

    Args:
        subs (str): substrate of interest
        screen_names (list, str): screens to parse for information
        path (str): path to data folder

    Returns:
        relevant_data (dict): dictionary with keys being screens with relevant data and values being pandas.core.series with relevant screening data
    """
    screen_data = {}
    substrates = {}
    count = 0

    # Read screening file and extract substrate lists for each
    for element in screen_names:
        name = 'screen_' + str(count)
        screen_data[name] = pd.read_csv(path + 'screening_data/%s' % element)
        substrates[name] = screen_data[name].Substrate.tolist()
        screen_data[name] = screen_data[name].set_index('Substrate')
        count += 1

    # Extract information relevant to protease and store in relevant_data
    relevant_data = {}
    for element in screen_data:
        if subs in substrates[element]:
            relevant_data[element] = screen_data[element].loc[subs]

    return relevant_data

def find_substrates(protease, screen, p_threshold):
    """ Apply thresholding to find substrates cleaved by protease of interest.

    Args:
        protease (str): protease of interest
        screen (str): path to screening file to be parsed
        p_threshold: adjusted p_value to threshold with

    Returns:
        cleaved_substrates (lst): list of csv substrates
    """

    raise NotImplementedError

def find_sequences(substrates, screen_name):
    """ Find sequences for substrates of interest

    Args:
        substrates (list, str): list of substrates of interest
        screen_name (str): screen where the substrates come from

    Returns:
        substrate_sequences (pandas.df): df with corresponding sequences
    """

    raise NotImplementedError

def make_logo(substrate_sequences, title, logo_format, size):
    """ Make SeqLogo plot for substrate_sequences

    Args:
        substrate_sequences (pandas.df): 1st col is substrate name, 2nd col is AA sequence
        title (str): title for SeqLogo
        logo_format (str): 'svg', 'eps', 'pdf', 'jpeg', 'png'
        size (str): 'small': 3.54" wide, 'medium': 5" wide, 'large': 7.25" wide, 'xlarge': 10.25" wide

    Returns:
        seqLogo
    """

    raise NotImplementedError