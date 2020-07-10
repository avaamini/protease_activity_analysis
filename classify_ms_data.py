import os
import protease_activity_analysis as paa
import pandas as pd
import argparse
import numpy as np

from utils import get_data_dir, get_output_dir, PLEX_14, RENAMED_14, \
    PLEX_20, RENAMED_20

parser = argparse.ArgumentParser()
parser.add_argument('--files', type=str, nargs="*",
    help='names of pickle files')
parser.add_argument('--pos_classes', type=str, default=None, nargs="*",
    help='names of positive classes')
parser.add_argument('--pos_class', type=str, default=None,
    help='name of positive class for re-labling')
parser.add_argument('--neg_classes', type=str, default=None, nargs="*",
    help='names of negative classes')
parser.add_argument('--neg_class', type=str, default=None,
    help='name of negative class for re-labling')
parser.add_argument('--class_type', type=str, default='svm', nargs="*",
    help='type of classifier: svm, random forest')
parser.add_argument('--num_folds', type=int, default=5,
    help='number of folds for cross validation')
parser.add_argument('--save', type=str, help='name to save plots')

if __name__ == '__main__':
    args = parser.parse_args()
    data_dir = get_data_dir()
    out_dir = get_output_dir()

    X, Y, _ = paa.data.make_class_dataset(data_dir, args.files,
        args.pos_classes, args.pos_class, args.neg_classes, args.neg_class)

    for classifier in args.class_type:
        probs, scores, tprs, aucs = paa.classify.classify_kfold_roc(X, Y,
            classifier, args.num_folds)
        save_name = args.save + "_" + classifier
        paa.vis.plot_kfold_roc(tprs, aucs, out_dir, save_name, show_sd=True)
