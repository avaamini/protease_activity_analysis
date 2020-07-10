import os
import protease_activity_analysis as paa
import argparse

from utils import get_data_dir, get_output_dir

# To run from terminal
# python test_kinetic.py --in_path="MMP13_stm.xlsx" --fc_time=45 --linear_time=15


parser = argparse.ArgumentParser()
parser.add_argument('--in_path', help='path to load data from')
parser.add_argument('--out_path', default=paa.tests.get_output_dir(), help='path to load data from')
parser.add_argument('--fc_time', type=int, help='time in min at which to take the fold change')
parser.add_argument('--linear_time', type=int, help='time in min to take initial speed')
args = parser.parse_args()


#Get data directory
data_dir = get_data_dir()
screen_path = os.path.join(data_dir, args.in_path)

[fc, fc_x, z_score_fc, init_rate, z_score_rate] = paa.vis.kinetic_analysis(in_path=screen_path, out_path=args.out_path, fc_time=args.fc_time, linear_time=args.linear_time)