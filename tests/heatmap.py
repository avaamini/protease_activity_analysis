import os
import protease_activity_analysis as paa
import argparse

from utils import get_data_dir, get_output_dir

# To run from terminal
# python heatmap.py --in_path="stm_kinetic/heatmap_fold_change_stm.xlsx" --out_path="heatmap.png"

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', help='path to load data from')
parser.add_argument('--out_path', help='path to load data from')
parser.add_argument('--metric', default='euclidean', help='Distance metric to use for the data')
parser.add_argument('--method', default='average', help='Linkage method for clustering')
parser.add_argument('--scale', default='log2', help='Scaling options')  #TODO: Include alternatives
args = parser.parse_args()


# Get data directory
data_dir = get_data_dir()
in_path = os.path.join(data_dir, args.in_path)
res_path = os.path.join(get_output_dir(), args.out_path)

# Plot heatmap
scaled = paa.vis.plot_heatmap(in_path, res_path)

