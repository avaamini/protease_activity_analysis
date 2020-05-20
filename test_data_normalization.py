import os

import protease_activity_analysis as paa

# test the data loading function
test_dir = paa.tests.get_data_dir()
data_path = os.path.join(test_dir, "Tx.01_3.5wks_Results.xlsx")
id_path = os.path.join(test_dir, "Tx.01_3.5wks_SampleType.xlsx")

sheets = ['Rev3-CONH2-1', 'Rev3-CONH2-2']
stock_name = "Stock"

syneos_data = paa.data.load_syneos(data_path, id_path, sheets, stock_name)

# print(syneos_data)

plex_14 = ["1UR3_01", "2UR3_02", "3UR3_03", "4UR3_04", "5UR3_05", "6UR3_06", 
           "7UR3_07", "8UR3_08", "9UR3_10", "1UR3_11", "2UR3_13", "3UR3_15", 
           "4UR3_16", "5UR3_18"]

plex_20 = []

normalized_matrix = paa.data.process_syneos_data(syneos_data, plex_14)

normalized_matrix.columns = plex_14
normalized_matrix.index = syneos_data.index[:-1]
normalized_matrix.reset_index(inplace=True)

update = paa.vis.plot_heatmap(normalized_matrix)

print("DONE!")

