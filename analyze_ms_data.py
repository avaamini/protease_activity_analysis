import os
import protease_activity_analysis as paa
import argparse

from utils import get_data_dir, get_output_dir

## TODO: change this function so that it does not take all arguments
def load_urine_data(args):
    data_dir = get_data_dir()

    # read syneos file
    syneos_data = paa.data.load_syneos(args.data_path, args.type_path, args.stock_path, args.sheets)

    # read plex/reporter file
    features, renamed = paa.data.get_plex(args.plex_path)

    # if only want to use a subset of the features to construct the data matrix
    if args.features_to_use != None:
        features = args.features_to_use

    # process the data and do normalizations
    filtered_data = paa.data.process_syneos_data(syneos_data, features,
        args.stock, args.type_filter, args.ID_filter, args.ID_exclude, args.save_name)
    mean_scaled = paa.data.mean_scale_matrix(filtered_data, args.save_name)
    z_scored = paa.data.standard_scale_matrix(filtered_data, args.save_name)

    mean_scaled.columns = renamed
    z_scored.columns = renamed # NOTE: this will result in feature-standardization

    # save data in pickle file
    pkl_name = args.pkl_name
    if pkl_name is not None:
        pkl_name_full = pkl_name + ".pkl"
        pkl_name_mean = pkl_name + "_mean" + ".pkl"
        pkl_name_z = pkl_name + "_zscore" + ".pkl"
        filtered_data.to_pickle(os.path.join(data_dir, pkl_name_full))
        mean_scaled.to_pickle(os.path.join(data_dir, pkl_name_mean))
        z_scored.to_pickle(os.path.join(data_dir, pkl_name_z))

    return mean_scaled, z_scored, plex, renamed, pkl_name, pkl_name_mean, pkl_name_z

if __name__ == '__main__':
    args = paa.parsing.parse_ms_args()
    out_dir = get_output_dir()

    mean_scaled, z_scored, plex, renamed, _, _, _ = load_urine_data(args)

    """ volcano plot """
    if args.volcano:
        paa.vis.plot_volcano(mean_scaled, args.group1, args.group2, renamed, out_dir,
            args.save_name)

    """ PCA """
    if args.pca:
        paa.vis.plot_pca(mean_scaled, renamed, args.pca_groups, out_dir, args.save_name)
