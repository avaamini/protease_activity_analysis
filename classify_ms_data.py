import os
import protease_activity_analysis as paa
import pandas as pd
import argparse
import numpy as np
from analyze_ms_data import load_urine_data

from utils import get_data_dir, get_output_dir

if __name__ == '__main__':
    args = paa.parsing.parse_ms_args()
    data_dir = get_data_dir()
    out_dir = get_output_dir()

    # supply a list of pkl files
    if args.files is not None:
        files = args.files
    else: # one file generated from reading in single excel file
        matrix, plex, renamed, pkl_name = load_urine_data(args)
        files = [pkl_name]

    # Multiclass classification
    if args.multi_class is not None:
        X, Y, _ = paa.data.make_multiclass_dataset(data_dir, files, args.multi_class)
        for classifier in args.class_type:
            file_name = args.save_name + "_" + classifier
            save_name = os.path.join(out_dir, file_name)
            probs, scores, cms = paa.classify.multiclass_classify(X, Y,
                classifier, args.num_folds, save_name)
    else: # Binary classification
        X, Y, _ = paa.data.make_class_dataset(data_dir, files,
            args.pos_classes, args.pos_class, args.neg_classes, args.neg_class)
        for classifier in args.class_type:
            probs, scores, tprs, aucs = paa.classify.classify_kfold_roc(X, Y,
                classifier, args.num_folds, args.pos_class)
            save_name = args.save_name + "_" + classifier
            paa.vis.plot_kfold_roc(tprs, aucs, out_dir, save_name, show_sd=True)

            # Recursive feature elimination analysis
            paa.classify.recursive_feature_elimination(X, Y, classifier,
                args.num_folds, out_dir, save_name)
