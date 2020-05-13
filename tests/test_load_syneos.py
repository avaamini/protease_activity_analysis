import os

import protease_activity_analysis as paa

# test the data loading function
test_dir = paa.tests.get_data_dir()
data_path = os.path.join(test_dir, "Tx.01_3.5wks_Results.xlsx")
id_path = os.path.join(test_dir, "Tx.01_3.5wks_SampleType.xlsx")

sheets = ['Rev3-CONH2-1', 'Rev3-CONH2-2']
stock_name = "Stock"

syneos_data = paa.data.load_syneos(data_path, id_path, sheets, stock_name)

print(syneos_data)
