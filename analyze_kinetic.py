import os
import pandas as pd
import protease_activity_analysis as paa
import argparse

from utils import get_data_dir, get_output_dir

# screen_path = os.path.join(data_dir, args.data_path)
# python analyze_kinetic.py --in_path='revitope/Final/MCA_AEBSF.xlsx' --fc_time=30 --linear_time=30

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
