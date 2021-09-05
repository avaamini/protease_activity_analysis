import os
import protease_activity_analysis as paa
import argparse

from utils import get_data_dir

data_dir = get_data_dir()
args = paa.parsing.parse_database_args()

data = paa.database.SubstrateDatabase(args.file_list)
q1_individual, q1_overall = data.get_top_hits('Q1', 'substrate', top_k=10, out_dir=None, z_threshold=None)
