import os
import protease_activity_analysis as paa
import argparse

from utils import get_data_dir

data_dir = get_data_dir()
args = paa.parsing.parse_database_args()

data = paa.database.SubstrateDatabase(
    data_files=args.data_files,
    sequence_file=args.sequence_file,
    names_file=args.names_file
)
q1_individual, q1_overall = data.get_top_hits('Q1', 'substrate', top_k=10, out_dir=None, z_threshold=None)

# python test_database.py --data_files data/screens/PAA/Bhatia1_PAA.csv data/s
# creens/PAA/Bhatia2_PAA.csv --sequence_file data/screens/PAA/Peptide_Inventory.csv --names_file data/screens/PAA/names_dict.pkl
