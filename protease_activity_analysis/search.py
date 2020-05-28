""" Collection of data searching/lookup functions based on existing screening data"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import ndtr

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


# for future: consider changing format of image or making it an argument + adding defaults + figure out bin width
def plot_distribution(screen_data, n_rows, n_cols, fig_size, col, title, x_label):
    """ Make subplot of screening data distributions (e.g. fold changes or z-scores)

    Args:
        screen_data (pandas.df): screening data with df.columns = proteases  and df.index=Substrates
        nrows (int): number of rows in subplot
        ncols (int): number of columns in subplot
        fig_size (tuple, int): fize of figure - e.g. '(width,height)' in inches
        col (str): color of plots e.g. 'g', r', 'b'
        title (str): title of subplot
        x_label (str): label of x-axis

    Returns:
        png: Subplot of distributions
    """
    # Plot distribution of fold changes
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=fig_size)
    fig.subplots_adjust(hspace=0.5)
    fig.subplots_adjust(wspace=0.5)
    sns.set(style="white", palette="muted", color_codes=True)

    for ax, feature in zip(axes.flatten(), screen_data):
        f = sns.distplot(screen_data[feature], ax=ax, color=col, axlabel=False,
                         kde=False)  # bins=len(np.unique(screen_data.T.iloc[0]))//2, color="g", axlabel = False)
        ax.set(title=feature)

    fig.suptitle(title)
    fig.text(0.5, 0.04, x_label, ha='center', va='center')
    fig.text(0.06, 0.5, 'Frequency', ha='center', va='center', rotation='vertical')
    fig.show()

    # plt.savefig(out_path, dpi=300)

    return


def plot_std(screen_data, n_bins=30):
    """ Compute std of proteases in raw data to threshold for inactivity

    Args:
        screen_data (pandas.df): screening data with df.columns = proteases  and df.index=Substrates
        n_bins (int): number of bins for histogram
    Returns:
        matplotlib figure: Histogram of std of raw data
        std_data (list): list of std of each column in the dataframe
    """
    # Compute std of proteases in raw data to identify those that did not cleave anything
    std_data = []

    for col in screen_data.columns:
        std_data.append(screen_data[col].std(ddof=0))

    plt.plot(figzise=(6, 4))
    plt.hist(std_data, bins=n_bins)
    plt.title('Histogram of Std of raw data')
    plt.xlabel('Raw Data Std')
    plt.ylabel('Frequency')
    plt.show()
    # plt.savefig('outputs/histogram_std_raw.png', dpi=300)

    # plt.savefig('outputs/histogram_std_raw.png', dpi=300)

    return std_data


def subset_active_inactive(screen_data, std_data, std_threshold):
    """ Subset screening data into active and inactive proteases

    Args:
        screen_data (pandas.df): screening data with df.columns = proteases  and df.index=Substrates
        std_data (list, float): list of std of each column in the dataframe, columns od screen_data nd std_data must match
        std_threshold (float): std threshold used for subsetting
    Returns:
        active_data (pandas.df): screening data from active proteases, above user set std threshold
        inactive_data (pandas.df): screening data from inactive proteases, below user set std threshold
    """
    # Add Std to dataframe of screen data
    std_data_df = pd.DataFrame({'Std': std_data}, index=screen_data.columns)
    screen_data = screen_data.append(std_data_df.T)

    # Subset into active and inactive
    active_data = screen_data.loc[:, screen_data.loc['Std'] > std_threshold]
    inactive_data = screen_data.loc[:, screen_data.loc['Std'] < std_threshold]

    return active_data, inactive_data


def calculate_zscore_pval(screen_data):
    """ Calculate z score by protease and calculate statistical significance by converting
    Z scores into p values by 1 - pnorm (Z value)

    Args:
        screen_data (pandas.df): screening data with df.columns = proteases  and df.index=Substrates

    Returns:
        screen_zscore (pandas.df): z-scored data
        screen_pval (pandas.df): p-values for data (note, not adjusted for mutiple hypothesis testing)
        min_pval (list, float): minimum pvalue for each protease
    """
    # Calculate Z score for only active
    first_col = screen_data.columns[0]
    dic = {first_col: []}
    screen_zscore = pd.DataFrame(dic)

    for col in screen_data.columns:
        screen_zscore[col] = ((screen_data[col] - screen_data[col].mean()) / screen_data[col].std(ddof=0))

    # Calculate p value of zscores for only active, keep a record of minimum p value fopr each protease
    screen_pval = pd.DataFrame(dic)
    min_pval = []

    for col in screen_zscore.columns:
        screen_pval[col] = 1 - ndtr(screen_zscore[col])
        min_pval.append(min(screen_pval[col]))

    return screen_zscore, screen_pval, min_pval
