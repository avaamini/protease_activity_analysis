import os
import pandas as pd
import protease_activity_analysis as paa

# Test search_protease and search_substrate
# Get directory for data folder
data_dir = paa.tests.get_data_dir()

# User-defined screens of interest
screens_of_interest = ['BhatiaScreen1_final.csv', 'BhatiaScreen2_final.csv', 'GlimpseSecreenFeb_final.csv', 'GlimpseSecreenDec_final.csv']

# Two examples
mmp1_df = paa.search.search_protease('MMP1', screens_of_interest, data_dir)

pq1_df = paa.search.search_substrate('PQ1', screens_of_interest, data_dir)

# Test screening data manipulation functions
screen_path = os.path.join(data_dir, 'screening_data/BhatiaScreen3_final.csv')
# out_path = '/Users/mariaalonso/Work/paa/protease_activity_analysis/tests/outputs'

# Import screening data
screen_data = pd.read_csv(screen_path)
screen_data = screen_data.set_index('Substrate')

# Plot distribution of fold changes
tit = 'Distributions of fold changes at end of LCS'
lab = 'Fold change'
paa.search.plot_distribution(screen_data, n_rows = 4, n_cols=7, fig_size = (14,8), col="g", title = tit, x_label=lab)


# Compute std of proteases in raw data to identify those that did not cleave anything
std_data = paa.search.plot_std(screen_data)
#plt.savefig('outputs/histogram_std_raw.png', dpi=300)

# Subset proteases into active and inactive based on std distribution
[active_data, inactive_data] = paa.search.subset_active_inactive(screen_data, std_data, std_threshold=0.1)

# Calculate Z score and pvalues for only active prtieases
active_zscore, active_pval, active_min_pval = paa.search.calculate_zscore_pval(active_data)

# Plot distribution of zscores
tit = 'Distributions of Zscores at end of LCS (Active proteases)'
lab = 'Z- score'
fig1 = paa.search.plot_distribution(active_zscore,n_rows = 5, n_cols=5, fig_size = (10,10), col="r", title = tit, x_label=lab)
# plt.savefig('outputs/histogram_zscores_active.png', dpi=300)
