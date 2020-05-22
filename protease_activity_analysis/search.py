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