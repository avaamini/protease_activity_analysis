#!/bin/bash
set -e

python analyze_kinetic.py --in_path="stm_kinetic/MMP13_stm.xlsx" --fc_time=45 --linear_time=15

python analyze_kinetic.py --in_path="stm_kinetic/PRSS3_stm.xlsx" --fc_time=45 --linear_time=15

python heatmap.py --in_path="stm_kinetic/heatmap_fold_change_stm.xlsx" --out_path="heatmap.png"

python analyze_ms_data.py --in_data_path="2017_12.19_BatchH_Lu.08_RESULTS.xlsx" --in_type_path="KP_7.5wks_IDtoSampleType.xlsx" -n Rev3-CONH2-1 Rev3-CONH2-2 --stock="inj" --num_plex=14 --ID_filter="2B" --save_name="KP_7-5"

python analyze_ms_data.py --in_data_path="2019_5.23_BatchLu.19-7.5 and Lu22 RESULTS.xlsx" --in_type_path="E_7.5wks_IDtoSampleType.xlsx" -n Rev3-CONH2-1 Rev3-CONH2-2 --stock="Stock-Lu19-7-5" --num_plex=14 --type_filter Control Eml4-Alk --save_name="EA_7-5"

python classify_ms_data.py  --files KP_7-5.pkl EA_7-5.pkl --pos_classes KP Eml4-Alk --pos_class LUAD --class_type svm rf --save LUAD_7-5
