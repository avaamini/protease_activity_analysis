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

    # supply a list of pkl files containing the data to train the classifier
    if args.files is not None:
        files = args.files
    else: # one file generated from reading in single excel file
        mean_scaled, z_scored, plex, renamed, pkl_name_mean, pkl_name_z = load_urine_data(args)
        if args.normalization == 'mean':
            files = [pkl_name_mean]
        if args.normalization == 'zscore':
            files = [pkl_name_z]

    # supply a list of pkl files constituing the independent test set
    if args.test is not None:
        test_files = args.test

    # Multiclass classification
    if args.multi_class is not None:
        X, Y, _ = paa.data.make_multiclass_dataset(data_dir, files, args.multi_class)

        # independent test set
        X_test = None
        Y_test = None
        if args.test is not None:
            X_test, Y_test, _ = paa.data.make_multiclass_dataset(data_dir,
                test_files, args.multi_class)

        for classifier in args.class_type:
            for kernel in args.kernel:
                classifier_name = classifier
                if classifier == 'svm':
                    classifier_name = classifier_name + "_" + kernel

                file_name = args.save_name + "_" + classifier_name
                save_name = os.path.join(out_dir, file_name)
                # set evaluation w/ cross-validation
                val_class_dict, test_class_dict = paa.classify.multiclass_classify(
                    X, Y, classifier, kernel, args.num_folds, save_name,
                    X_test, Y_test)

    else: # Binary classification
        # k fold cross validation
        X, Y, _ = paa.data.make_class_dataset(data_dir, files,
            args.pos_classes, args.pos_class, args.neg_classes, args.neg_class)

        # independent test set
        X_test = None
        Y_test = None
        if args.test is not None:
            X_test, Y_test, _ = paa.data.make_class_dataset(data_dir, test_files,
                args.pos_classes, args.pos_class, args.neg_classes, args.neg_class)

        for classifier in args.class_type:
            for kernel in args.kernel:
                classifier_name = classifier
                if classifier == 'svm':
                    classifier_name = classifier_name + "_" + kernel

                # evaluation w/ cross-validation
                val_class_dict, test_class_dict = paa.classify.classify_kfold_roc(
                    X, Y, classifier, kernel, args.num_folds, args.pos_class,
                    X_test, Y_test)

                # cross-validation performance
                tprs_val = val_class_dict["tprs"]
                aucs_val = val_class_dict["aucs"]

                save_name_val = args.save_name + "_crossval_" + classifier_name
                paa.vis.plot_kfold_roc(tprs_val, aucs_val,
                    out_dir, save_name_val, show_sd=True)

                # independent test set performance
                if args.test is not None:
                    tprs_test = test_class_dict["tprs"]
                    aucs_test = test_class_dict["aucs"]

                    save_name_test = args.save_name + "_test_" + classifier_name
                    paa.vis.plot_kfold_roc(tprs_test, aucs_test,
                        out_dir, save_name_test, show_sd=True)


                # Recursive feature elimination analysis ONLY with rf, lr, linear svm
                if classifier == "svm" and kernel != "linear":
                    break
                paa.classify.recursive_feature_elimination(X, Y,
                    classifier, kernel, args.num_folds, out_dir, save_name_val)
