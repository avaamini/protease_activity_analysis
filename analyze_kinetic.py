import os
import protease_activity_analysis as paa
import argparse

from utils import get_data_dir, get_output_dir

args = paa.parsing.parse_kinetic_args()
data_dir = get_data_dir()
out_dir = get_output_dir()

screen_path = os.path.join(data_dir, args.data_path)

[fc, fc_x, z_score_fc, init_rate, z_score_rate] = paa.vis.kinetic_analysis(
    in_path=screen_path, out_path=out_dir, fc_time=args.fc_time,
    linear_time=args.linear_time)
