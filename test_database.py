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
# Should return error because can't search by Q1 (yet), S52 or something from the screens that are input should work
q1_individual, q1_overall = data.get_top_hits('Q1', 'substrate', top_k=10, out_dir=None, z_threshold=None)
#Test
# python test_database.py --data_files data/screens/PAA/PAA_screens/Bhatia1_PAA.csv data/screens/PAA/PAA_screens/Bhatia2_PAA.csv --sequence_file data/screens/PAA/Peptide_Inventory_150.csv --names_file data/screens/PAA/names_dict.pkl
### For kinetic screen and sequence info files, need to have the substrate names by indicated by 'Name' (rather than 'PAA')
### this used to be the case, but i just fixed it to just take the name of the first column (rather than 'Name'), for greater modularity
