""" Argument parsing for interfacing with command line. """
from argparse import ArgumentParser, Namespace

def add_ms_args(parser: ArgumentParser):
    """ Add arguments for MS data analysis.

    Args:
        parser: An ArgumentParser.
    """

    ## File and data arguments
    parser.add_argument('--files', type=str, nargs="*", default=None,
        help='names of pickle files for analysis/training the classifier')
    parser.add_argument('--test_files', type=str, nargs="*", default=None,
        help='names of pickle files to specify independent test data')
    parser.add_argument('--data_path', type=str, default=None,
        help='path to load data from')
    parser.add_argument('--type_path', type=str, default=None,
        help='path to load Sample Types from')
    parser.add_argument('--plex_path', type=str, default=None,
        help='file that contains reporter/plex nomenclature')
    parser.add_argument('--stock_path', type=str, default=None,
        help='file that contains sample/stock pairs for normalization. \
            if not specified, assumes only one stock by default.')
    parser.add_argument('--stock', type=str, nargs="*", default=None,
        help='name of the stocks for normalization. can specify multiple.')
    parser.add_argument('-n', '--sheets', type=str, nargs="*", default=None,
        help='number of sheets for excel file')

    ## Filter, group, and label arguments
    # Data filtering
    parser.add_argument('--type_include', type=str, nargs="*", default=None,
        help='nomenclature filter for sample type to use (keep)')
    parser.add_argument('--ID_include', type=str, default=None,
        help='nomenclature filter for sample ID to use (keep)')
    parser.add_argument('--features_include', type=str, nargs="*", default=None,
        help='names of features to include in analysis')
    parser.add_argument('--ID_exclude', type=str, nargs='*', default=None,
        help='IDs for samples to exclude from the output matrix')

    # Volcano/PCA filtering
    parser.add_argument('--group1', type=str, nargs='*', default=None,
        help='first set of sample types for significance comparison in volcano')
    parser.add_argument('--group2', type=str, nargs='*', default=None,
        help='second set of sample types for significance comparison in volcano')
    parser.add_argument('--pca_groups', type=str, nargs='*', default=None,
        help='specify sample types for consideration in PCA')
    parser.add_argument('--group_key', type=str, default='Sample Type',
        help='sample annotation by which to annotate the PCA, i.e., sample type')
    parser.add_argument('--biplot', type=bool, default=False,
        help='True if a biplot annotation on PCA is desired')

    ## Classification arguments
    parser.add_argument('--multi_class', type=str, nargs='*', default=None,
        help='names of classes for multi class classification')
    parser.add_argument('--pos_classes', type=str, nargs="*", default=None,
        help='names of positive classes')
    parser.add_argument('--pos_class', type=str, default=None,
        help='name of positive class for re-labling')
    parser.add_argument('--neg_classes', type=str, nargs="*", default=None,
        help='names of negative classes')
    parser.add_argument('--neg_class', type=str, default=None,
        help='name of negative class for re-labling')
    parser.add_argument('--test_types', type=str, nargs="*", default=None,
        help='names of sample types to hold out from training; test types')
    parser.add_argument('--scale', type=bool, default=False,
        help='whether to apply feature scaling/standardization for classification')
    parser.add_argument('--seed', type=int, default=None,
        help='random integer to set the random state of the classifer.')
    parser.add_argument('--class_type', type=str, nargs="*", default=['svm'],
        help='type of classifier: svm, rf (random forest), lr (logistic regression)')
    parser.add_argument('--kernel', type=str, nargs="*", default=['linear'],
        help='type of kernel for svm: linear, rbf, poly')
    parser.add_argument('--num_folds', type=int, default=5,
        help='number of folds for cross validation')
    parser.add_argument('--val_split', type=float, default=0.2,
        help='percent of training set for validation')

    ## Analysis arguments
    parser.add_argument('--normalization', type=str, default=None,
        help='type of normalization to use for classification')
    parser.add_argument('--volcano', action='store_true', default=False,
        help='use to plot volcano plots')
    parser.add_argument('--pca', action='store_true', default=False,
        help='use to plot PCA')

    ## Save arguments
    parser.add_argument('--save_name', type=str,
        help='name to save plots and files')
    parser.add_argument('--save_dir', type=str, default='outputs',
        help='path to directory for saving plots and files')

def parse_ms_args() -> Namespace:
    """ Parse MS data analysis arguments."""
    parser = ArgumentParser()
    add_ms_args(parser)
    args = parser.parse_args()
    return args

def add_kinetic_args(parser: ArgumentParser):
    """ Add arguments for kinetic data analysis.

    Args:
        parse: ArgumentParser
    """
    parser.add_argument('--data_path', type=str, default=None,
        help='path to load data from')
    parser.add_argument('--fc_time', type=int, default=30,
        help='time in min at which to take the fold change')
    parser.add_argument('--linear_time', type=int, default=30,
        help='time in min to take initial speed')

def parse_kinetic_args() -> Namespace:
    """ Parse kinetic data analysis arguments."""
    parser = ArgumentParser()
    add_kinetic_args(parser)
    args = parser.parse_args()
    return args

def add_database_args(parser: ArgumentParser):
    """ Add arguments for constructing the substrate database.

    Args:
        parser: ArgumentParser
    """
    parser.add_argument('--data_files', type=str, nargs="*", default=None,
        help='list of dataset files to load into the database')
    parser.add_argument('--sequence_file', type=str, default=None,
        help='path to file containing the substrate-sequence information')
    parser.add_argument('--names_file', type=str, default=None,
        help='path to file containing substrate names and descriptors')

def parse_database_args() -> Namespace:
    """ Parse database arguments."""
    parser = ArgumentParser()
    add_database_args(parser)
    args = parser.parse_args()
    return args
