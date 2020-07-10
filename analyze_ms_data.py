import os
import protease_activity_analysis as paa
import argparse

from utils import get_data_dir, get_output_dir, PLEX_14, RENAMED_14, \
    PLEX_20, RENAMED_20

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
parser.add_argument('--group1', default=None, type=str,
    help='first group for significance comparison')
parser.add_argument('--group2', default=None, type=str,
    help='second group for significance comparison')
parser.add_argument('--save_name', type=str, help='name to save')

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
