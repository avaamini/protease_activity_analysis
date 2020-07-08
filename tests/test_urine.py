import os
import protease_activity_analysis as paa
import argparse

# To run: STM 7.5wk data
# python tests/test_urine_volcano.py --in_data_path="2017_12.19_BatchH_Lu.08_RESULTS.xlsx" --in_type_path="KP_7.5wks_IDtoSampleType.xlsx" -n Rev3-CONH2-1 Rev3-CONH2-2 --stock="inj" --num_plex=14 --ID_filter="2B" --group1="Control" --group2="KP"

parser = argparse.ArgumentParser()
parser.add_argument('--in_data_path', help='path to load data from')
parser.add_argument('--in_type_path', help='path to load Sample Types from')
parser.add_argument('-n', '--sheets', type=str, nargs="*")
parser.add_argument('--stock', help='name of the stock sample in the Inventiv file')
parser.add_argument('--num_plex', type=int, help='number of reporters (14 or 20)')
parser.add_argument('--type_filter', default=None, type=str,
    help='nomenclature filter for sample type to use')
parser.add_argument('--ID_filter', default=None, type=str,
    help='nomenclature filter for sample ID to use')
parser.add_argument('--group1', type=str)
parser.add_argument('--group2', type=str)
args = parser.parse_args()

data_dir = paa.tests.get_data_dir()
data_path = os.path.join(data_dir, args.in_data_path)
id_path = os.path.join(data_dir, args.in_type_path)

syneos_data = paa.data.load_syneos(data_path, id_path, args.sheets, args.ID_filter)

# 14-plex
if args.num_plex == 14:
    plex = ["1UR3_01", "2UR3_02", "3UR3_03", "4UR3_04", "5UR3_05", "6UR3_06",
           "7UR3_07", "8UR3_08", "9UR3_10", "1UR3_11", "2UR3_13", "3UR3_15",
           "4UR3_16", "5UR3_18"]
    renamed = ["PP01", "PP02", "PP03", "PP04", "PP05", "PP06", "PP07", "PP08",
              "PP09", "PP10", "PP11", "PP12", "PP13", "PP14"]
elif args.num_plex == 20:
## 20-plex
    plex= ["1UR3_01", "2UR3_02", "3UR3_03", "4UR3_04", "5UR3_05", "6UR3_06",
                "7UR3_07","8UR3_08", "9UR3_10", "1UR3_11", "2UR3_13", "3UR3_15",
                "4UR3_16", "5UR3_18", "6UR3_20", "1UR3_09", "2UR3_12", "3UR3_14",
                "4UR3_17", "5UR3_19"]
    renamed = ["PP01", "PP02", "PP03", "PP04", "PP05", "PP06", "PP07", "PP08",
                  "PP09", "PP10", "PP11", "PP12", "PP13", "PP14", "PP15", "PP16",
                  "PP17", "PP18", "PP19", "PP20"]

syneos_data = syneos_data.drop(args.stock, level='Sample ID')
normalized_matrix = paa.data.process_syneos_data(syneos_data, plex, args.ID_filter)
normalized_matrix.columns = renamed

""" to create volcano plots """
paa.vis.plot_volcano(normalized_matrix, args.group1, args.group2, renamed, data_dir)

""" to run PCA """
undo_multiindex = normalized_matrix.reset_index()

if args.type_filter != None:
	undo_multiindex = undo_multiindex[~undo_multiindex['Sample Type'].str.contains(args.type_filter)]

paa.vis.plot_pca(undo_multiindex, renamed, data_dir)
