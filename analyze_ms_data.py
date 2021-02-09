import os
import protease_activity_analysis as paa
import argparse

from utils import get_data_dir, get_output_dir

## TODO: change this function so that it does not take all arguments
def load_urine_data(args):
    data_dir = get_data_dir()

    # read syneos file
    syneos_data = paa.data.load_syneos(args.data_path, args.type_path, args.sheets)

    # read plex/reporter file
    plex, renamed = paa.data.get_plex(args.plex_path)

    # process the data and do normalizations
    normalized_matrix = paa.data.process_syneos_data(syneos_data, plex,
        args.stock, args.type_filter, args.ID_filter, args.ID_exclude, args.save_name)
    normalized_matrix.columns = renamed

    # save data in pickle file
    if args.pkl_name is not None:
        pkl_name = args.pkl_name + ".pkl"
        normalized_matrix.to_pickle(os.path.join(data_dir, pkl_name))
    else:
        pkl_name = args.pkl_name

    return normalized_matrix, plex, renamed, pkl_name

if __name__ == '__main__':
    args = paa.parsing.parse_ms_args()
    out_dir = get_output_dir()

    matrix, plex, renamed, _ = load_urine_data(args)

    """ volcano plot """
    if args.volcano:
        paa.vis.plot_volcano(matrix, args.group1, args.group2, renamed, out_dir,
            args.save_name)

    """ PCA """
    if args.pca:
        paa.vis.plot_pca(matrix, renamed, args.pca_groups, out_dir, args.save_name)
