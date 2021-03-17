import os

import protease_activity_analysis as paa

# test the data loading function
test_dir = paa.tests.get_data_dir()
data_path = os.path.join(test_dir, "BV.03_Inventiv_Batch 2_RESULTS.xlsx")
id_path = os.path.join(test_dir, "BV.03_IDtoSampleType_Batch2.xlsx")

sheets = ['Rev3-CONH2-1', 'Rev3-CONH2-2']
stock_name = "Stock"

syneos_data = paa.data.load_syneos(data_path, id_path, sheets, stock_name)

# 14-plex
plex = ["1UR3_01", "2UR3_02", "3UR3_03", "4UR3_04", "5UR3_05", "6UR3_06", 
           "7UR3_07", "8UR3_08", "9UR3_10", "1UR3_11", "2UR3_13", "3UR3_15", 
           "4UR3_16", "5UR3_18"]

renamed = ["PP01", "PP02", "PP03", "PP04", "PP05", "PP06", "PP07", "PP08",
              "PP09", "PP10", "PP11", "PP12", "PP13", "PP14"]

## 20-plex
# plex= ["1UR3_01", "2UR3_02", "3UR3_03", "4UR3_04", "5UR3_05", "6UR3_06", 
#            "7UR3_07","8UR3_08", "9UR3_10", "1UR3_11", "2UR3_13", "3UR3_15", 
#            "4UR3_16", "5UR3_18", "6UR3_20", "1UR3_09", "2UR3_12", "3UR3_14", 
#            "4UR3_17", "5UR3_19"]
# renamed = ["PP01", "PP02", "PP03", "PP04", "PP05", "PP06", "PP07", "PP08",
#               "PP09", "PP10", "PP11", "PP12", "PP13", "PP14", "PP15", "PP16", 
#               "PP17", "PP18", "PP19", "PP20"]

syneos_data = syneos_data[:-1]
normalized_matrix = paa.data.process_syneos_data(syneos_data, plex)
normalized_matrix.index = syneos_data.index
normalized_matrix.columns = renamed


# """ to make a heatmap """
# update = paa.vis.plot_heatmap(normalized_matrix, renamed)
  
""" to perform PCA """
undo_multiindex = normalized_matrix.reset_index()

#if you want to filter for only some Sample Types, change code below
#undo_multiindex = undo_multiindex [~undo_multiindex['Sample Type'].str.contains("LAM")]
  
pca = paa.vis.plot_pca(undo_multiindex, renamed)

# """ to create volcano plots """
# volcano = paa.vis.plot_volcano(normalized_matrix, "Control", "S", renamed)
