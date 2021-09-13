import os
import protease_activity_analysis as paa
import pandas as pd
import argparse
import numpy as np

args = paa.parsing.parse_ms_args()

# supply a list of pkl files containing the data to train the classifier
if args.files is not None:
    files = args.files
else: # one file generated from reading in single excel file
    """ Load data. """
    syneos_dataset = paa.syneos.SyneosDataset(
        save_dir=args.save_dir, save_name=args.save_name)
    syneos_dataset.read_syneos_data(
        args.data_path, args.type_path, args.stock_path, args.sheets)

    # read plex/reporter file
    features, renamed = syneos_dataset.set_feature_mapping(args.plex_path)

    # if only want to use a subset of the features to construct the data matrix
    if args.features_include != None:
        features = args.features_include

    """ Process and normalizations. """
    syneos_dataset.process_syneos_data(
        features,
        args.stock,
        args.type_include,
        args.ID_include,
        args.ID_exclude
    )

    syneos_dataset.mean_scale_matrix()
    syneos_dataset.standard_scale_matrix()

    # write data to pickle files
    syneos_dataset.data_to_pkl(args.save_name)

    if args.normalization == 'mean':
        files = [f"{args.save_name}_mean.pkl"]
    else:
        files = [f"{args.save_name}.pkl"]

data_for_class = paa.syneos.SyneosDataset(
    save_dir=args.save_dir, save_name=args.save_name, file_list=files)

independent_test = (args.test_files is not None) or (args.test_types is not None)

""" classification. """
if args.multi_class is not None:

    X, Y, df, X_test, Y_test, df_test = data_for_class.make_multiclass_dataset(
        args.multi_class, args.test_types
    )

    if args.test_files is not None:
        test_data = paa.syneos.SyneosDataset(
            save_dir=args.save_dir,
            save_name=f"{args.save_name}_test",
            file_list=args.test_files
        )

        X_test, Y_test, df_test, _, _, _ = test_data.make_multiclass_dataset(
            args.multi_class, args.test_types
        )

    for classifier in args.class_type:
        for kernel in args.kernel:
            classifier_name = classifier
            if classifier == 'svm':
                classifier_name = classifier_name + "_" + kernel

            file_name = args.save_name + "_" + classifier_name
            save_name = os.path.join(args.save_dir, file_name)
            # set evaluation w/ cross-validation
            val_class_dict, val_df, test_class_dict, test_df = \
                paa.classify.multiclass_classify(
                    X, Y, classifier, kernel, args.num_folds, save_name,
                    args.scale, args.seed, X_test, Y_test
                )

            classes = np.unique(Y)

            # plot confusion matrix
            save_name_val = args.save_name + "_crossval_" + classifier_name
            paa.vis.plot_confusion_matrix(val_df, classes, classes, args.save_dir,
                save_name_val, cmap='Purples')

            if independent_test:
                test_classes = np.unique(Y_test)
                save_name_test = args.save_name + "_test_" + classifier_name
                paa.vis.plot_confusion_matrix(test_df, classes, test_classes,
                    args.save_dir, save_name_test, cmap='Greens')

else: # Binary classification with k fold cross validation

    X, Y, df, X_test, Y_test, df_test = data_for_class.make_class_dataset(
        args.pos_classes,
        args.pos_class,
        args.neg_classes,
        args.neg_class,
        args.test_types
    )

    if args.test_files is not None:
        test_data = paa.syneos.SyneosDataset(
            save_dir=args.save_dir,
            save_name=f"{args.save_name}_test",
            file_list=args.test_files
        )

        X_test, Y_test, df_test, _, _, _ = test_data.make_class_dataset(
            args.pos_classes,
            args.pos_class,
            args.neg_classes,
            args.neg_class,
            args.test_types
        )

    for classifier in args.class_type:
        for kernel in args.kernel:
            classifier_name = classifier
            if classifier == 'svm':
                classifier_name = classifier_name + "_" + kernel

            # evaluation w/ cross-validation
            val_class_dict, test_class_dict = paa.classify.classify_kfold_roc(
                X, Y, classifier, kernel, args.num_folds, args.pos_class,
                args.scale, args.seed,
                X_test, Y_test)

            # cross-validation performance
            tprs_val = val_class_dict["tprs"]
            aucs_val = val_class_dict["aucs"]

            save_name_val = args.save_name + "_crossval_" + classifier_name
            paa.vis.plot_kfold_roc(tprs_val, aucs_val,
                args.save_dir, save_name_val, show_sd=True)

            # independent test set performance
            if independent_test:
                tprs_test = test_class_dict["tprs"]
                aucs_test = test_class_dict["aucs"]

                save_name_test = args.save_name + "_test_" + classifier_name
                paa.vis.plot_kfold_roc(tprs_test, aucs_test,
                    args.save_dir, save_name_test, show_sd=True)

            # recursive feature elimination -- ONLY with rf, lr, svm linear!
            if classifier == 'svm' and kernel != 'linear':
                break
            paa.classify.rfe_cv(X, Y, classifier, args.num_folds,
                args.save_dir, save_name_val)
