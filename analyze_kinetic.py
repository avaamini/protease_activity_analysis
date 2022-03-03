import os
import pandas as pd
import protease_activity_analysis as paa
import argparse

from utils import get_data_dir, get_output_dir

""" Script to analyze and visualize kinetic protease activity data."""

if __name__ == '__main__':
    data_dir = get_data_dir()
    out_dir = get_output_dir()

    args = paa.parsing.parse_kinetic_args()

    data = paa.kinetic.KineticDataset(
        args.data_path,
        args.fc_time,
        args.linear_time,
        out_dir
    )

    data.plot_kinetic(
        kinetic_data = data.fc,
        title = data.sample_name,
        ylabel = 'FoldChange'
    )

    data.plot_kinetic(
        kinetic_data = data.raw,
        title = data.sample_name,
        ylabel = 'RawIntensity'
    )

    data.write_csv(
         data_to_write=data.fc_x,
         save_name='fc'
     )
