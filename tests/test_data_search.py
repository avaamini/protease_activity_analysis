import os

import protease_activity_analysis as paa

# Get directory for data folder
data_dir = paa.tests.get_data_dir()

# User-defined screens of interest
screens_of_interest = ['BhatiaScreen1_final.csv', 'BhatiaScreen2_final.csv', 'GlimpseSecreenFeb_final.csv', 'GlimpseSecreenDec_final.csv']

# Two examples
mmp1_df = paa.search.search_protease('MMP1', screens_of_interest, data_dir)

pq1_df = paa.search.search_substrate('PQ1', screens_of_interest, data_dir)
