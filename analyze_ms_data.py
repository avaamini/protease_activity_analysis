import os
import protease_activity_analysis as paa
import argparse

args = paa.parsing.parse_ms_args()

""" Read data file and create data loader. """
syneos_dataset = paa.syneos.SyneosDataset(
    save_dir=args.save_dir, save_name=args.save_name)
syneos_dataset.read_syneos_data(
    args.data_path, args.type_path, args.stock_path, args.sheets)

# read plex/reporter file
features, renamed = syneos_dataset.set_feature_mapping(args.plex_path)

# if only want to use a subset of the features to construct the data matrix
if args.features_include != None:
    features = args.features_include

""" Process and normalizations. """
syneos_dataset.process_syneos_data(
    features,
    args.stock,
    args.type_include,
    args.ID_include,
    args.ID_exclude
)
syneos_dataset.mean_scale_matrix()
syneos_dataset.standard_scale_matrix()

# write data to pickle files
syneos_dataset.data_to_pkl(args.save_name)

""" volcano plot """
if args.volcano:
    paa.vis.plot_volcano(
        syneos_dataset.mean_scaled_matrix,
        syneos_dataset.features,
        args.group_key,
        args.group1,
        args.group2,
        args.save_dir,
        args.save_name
    )

""" PCA """
if args.pca:
    paa.vis.plot_pca(
        syneos_dataset.mean_scaled_matrix,
        syneos_dataset.features,
        args.group_key,
        args.pca_groups,
        args.biplot,
        args.save_dir,
        args.save_name
    )
