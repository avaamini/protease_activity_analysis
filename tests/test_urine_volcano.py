import os
import protease_activity_analysis as paa
import argparse

# To run: 
# python tests/test_urine_volcano.py --in_data_path="2019_11.21_BatchBV.01 RESULTS.xlsx" --in_type_path="BV.01_IDtoSampleType.xlsx" -n Rev3-CONH2-1 Rev3-CONH2-2 --stock="Stock" --num_plex=14 --group1="Control" --group2="S"

parser = argparse.ArgumentParser()
parser.add_argument('--in_data_path', help='path to load data from')
parser.add_argument('--in_type_path', help='path to load Sample Types from')
parser.add_argument('-n', '--sheets', type=str, nargs="*")
parser.add_argument('--stock', help='name of the stock sample in the Inventiv file')
parser.add_argument('--num_plex', type=int, help='number of reporters (14 or 20)')
parser.add_argument('--group1', type=str)
parser.add_argument('--group2', type=str)
args = parser.parse_args()

data_dir = paa.tests.get_data_dir()
data_path = os.path.join(data_dir, args.in_data_path)
id_path = os.path.join(data_dir, args.in_type_path)

# test the data loading function
# test_dir = paa.tests.get_data_dir()
# data_path = os.path.join(test_dir, "2019_11.21_BatchBV.01 RESULTS.xlsx")
# id_path = os.path.join(test_dir, "BV.01_IDtoSampleType.xlsx")
# sheets = ['Rev3-CONH2-1', 'Rev3-CONH2-2']
# stock_name = "Stock"

syneos_data = paa.data.load_syneos(data_path, id_path, args.sheets, args.stock)

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
normalized_matrix = paa.data.process_syneos_data(syneos_data, plex)
normalized_matrix.index = syneos_data.index
normalized_matrix.columns = renamed

""" to create volcano plots """
volcano = paa.vis.plot_volcano(normalized_matrix, args.group1, args.group2, renamed, data_dir)

