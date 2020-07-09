import os
import protease_activity_analysis as paa
import argparse

## Global variables: reporter names for 14 and 20 plex MS panels
PLEX_14 = ["1UR3_01", "2UR3_02", "3UR3_03", "4UR3_04", "5UR3_05", "6UR3_06",
       "7UR3_07", "8UR3_08", "9UR3_10", "1UR3_11", "2UR3_13", "3UR3_15",
       "4UR3_16", "5UR3_18"]
RENAMED_14 = ["PP01", "PP02", "PP03", "PP04", "PP05", "PP06", "PP07", "PP08",
          "PP09", "PP10", "PP11", "PP12", "PP13", "PP14"]

PLEX_20 = ["1UR3_01", "2UR3_02", "3UR3_03", "4UR3_04", "5UR3_05", "6UR3_06",
            "7UR3_07","8UR3_08", "9UR3_10", "1UR3_11", "2UR3_13", "3UR3_15",
            "4UR3_16", "5UR3_18", "6UR3_20", "1UR3_09", "2UR3_12", "3UR3_14",
            "4UR3_17", "5UR3_19"]
RENAMED_20 = ["PP01", "PP02", "PP03", "PP04", "PP05", "PP06", "PP07", "PP08",
              "PP09", "PP10", "PP11", "PP12", "PP13", "PP14", "PP15", "PP16",
              "PP17", "PP18", "PP19", "PP20"]

parser = argparse.ArgumentParser()
parser.add_argument('--in_data_path', help='path to load data from')
parser.add_argument('--in_type_path', help='path to load Sample Types from')
parser.add_argument('-n', '--sheets', type=str, nargs="*")
parser.add_argument('--stock', help='name of the stock in the Inventiv file')
parser.add_argument('--num_plex', type=int, help='number reporters (14 or 20)')
parser.add_argument('--type_filter', default=None, type=str, nargs="*",
    help='nomenclature filter for sample type to use')
parser.add_argument('--ID_filter', default=None, type=str,
    help='nomenclature filter for sample ID to use')
parser.add_argument('--group1', type=str)
parser.add_argument('--group2', type=str)
parser.add_argument('--save_name', type=str, help='name to save')

def get_data_dir():
    test_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(test_dir, "data")

def get_output_dir():
    test_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(test_dir, "outputs")

def load_urine_data(data_dir, args):
    # get data files
    data_path = os.path.join(data_dir, args.in_data_path)
    id_path = os.path.join(data_dir, args.in_type_path)

    # read syneos file
    syneos_data = paa.data.load_syneos(data_path, id_path, args.sheets)

    # 14-plex
    if args.num_plex == 14:
        plex = PLEX_14
        renamed = RENAMED_14
    elif args.num_plex == 20:
    ## 20-plex
        plex = PLEX_20
        renamed = RENAMED_20

    # process the data and do normalizations
    normalized_matrix = paa.data.process_syneos_data(syneos_data, plex,
        args.stock, args.type_filter, args.ID_filter)
    normalized_matrix.columns = renamed

    # save data in pickle file
    matrix_name = args.save_name + ".pkl"
    normalized_matrix.to_pickle(os.path.join(data_dir, matrix_name))

    return normalized_matrix, plex, renamed

if __name__ == '__main__':
    args = parser.parse_args()
    data_dir = get_data_dir()
    out_dir = get_output_dir()

    matrix, plex, renamed = load_urine_data(data_dir, args)

    """ volcano plot """
    paa.vis.plot_volcano(matrix, args.group1, args.group2, renamed, out_dir,
        args.save_name)

    """ PCA """
    paa.vis.plot_pca(matrix, renamed, out_dir, args.save_name)
