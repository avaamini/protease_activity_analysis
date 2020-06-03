import os
import pandas as pd
import protease_activity_analysis as paa

# Get directory for data folder
data_dir = paa.tests.get_data_dir()

# Test screening data manipulation functions
data_path = os.path.join(data_dir, 'proteases_by_family_May_2020.xlsx')
sheet_names = ['Metallo', 'Cysteine', 'Aspartic','Serine']

protease_names = paa.data.read_names_from_uniprot(data_path,  sheet_names)

print('# Metallo protease names:', len(protease_names['Metallo']))
print('# Serine protease names:', len(protease_names['Serine']))
print('# Aspartic protease names:', len(protease_names['Aspartic']))
print('# Cysteine protease names:', len(protease_names['Cysteine']))

names_metallo=pd.DataFrame(protease_names['Metallo'], columns = ['Metallo'])
names_serine=pd.DataFrame(protease_names['Serine'], columns = ['Serine'])
names_aspartic=pd.DataFrame(protease_names['Aspartic'], columns = ['Aspartic'])
names_cysteine=pd.DataFrame(protease_names['Cysteine'], columns = ['Cysteine'])

# names_metallo.to_csv('outputs/names_metallo.csv')
# names_serine.to_csv('outputs/names_serine.csv')
# names_aspartic.to_csv('outputs/names_aspartic.csv')
# names_cysteine.to_csv('outputs/names_cysteine.csv')