""" Sequence-related functions."""
import pandas as pd
import scipy as sp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import scipy.stats as ss
import csv
import pickle
import protease_activity_analysis as paa

from scipy import stats
from scipy.stats import zscore
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from colorama import init, Fore, Back, Style
from fuzzywuzzy import fuzz

def color_seq(ex_sub, all_natural, aa_dict):
    """ Returns AA sequence color-coded by RasMol color scheme

        Args:
            ex_sub (str): AA sequence to color code
            all_natural (bool): if True color-code, if False return regular sequence
            aa_dict (dictionary): aa dictionary with color scheme to use for each
                one-letter AA
        Returns:
            color_sub (str): color-coded AA sequence
    """
    len_ex_sub = len(ex_sub)

    if all_natural:
        color_sub_list = [aa_dict[char] for char in ex_sub]
        color_sub = "".join(color_sub_list)

    else:  # contains un-natural amino acids. reset to standard color
        print(Style.RESET_ALL)
        color_sub = ex_sub

    return color_sub

def generate_kmers(sub_list, seq_list, k):
    """ Generate dictionary containing kmers of length k in each substrate
    sequence provided.

        Args:
            sub_list (list, str): list of substrate names
            seq_list (list, str): list of sequences
            k (int): kmer length
        Returns:
            kmer_dict (dict): dictionary where keys are substrate names.
                values are kmers of length k in a given substrate.
    """

    # dictionary where keys are substrate names and values are kmers
    kmer_dict = {}
    for i in np.arange(len(sub_list)):
        temp_seq = seq_list[i]
        n = len(temp_seq) - k + 1
        temp_kmer = []
        for j in np.arange(n): temp_kmer.append(temp_seq[j:j + k])
        kmer_dict[sub_list[i]] = temp_kmer

    return kmer_dict

def find_overlapping_kmers(kmer_dict):
    """ Returns dictionary where keys are kmers and values are substrates
    containing each kmer.

        Args:
            kmer_dict (dict): dictionary where keys are substrate names.
                values are kmers of length k in a given substrate
        Returns:
            overlap_dict (dict): dictionary where keys are kmer sequences.
                values are substrates containg a given kmer.
    """
    unique = list(sorted({ele for val in kmer_dict.values() for ele in val}))
    print('Number of unique kmers is:', len(unique))

    # create new dictionary that will store peptides with overlapping kmers
    overlap_dict = {}
    overlap_keys = unique
    for i in overlap_keys:
        overlap_dict[i] = []

    for el in unique:
        for key in kmer_dict.keys():
            if el in kmer_dict[key]:
                overlap_dict[el].append(key)

    return overlap_dict

def search_kmer(kmer_q, kmer_dict_q):
    """ Search kmer of length k in a given dictionary, kmer_dict_q, to return
    substrates containing that kmer.

        Args:
            kmer_q (str): kmer to query
            kmer_dict_q (dictionary): dictionary containing kmer breakdown, of
                kmers mapping to substrates.
        Returns:
            kmer_df (pandas df): dataframe of substrates containing the query
                kmer.
        Raises:
            KeyError: if query kmer not in dataset dictionary
    """
    keys_q = kmer_dict_q.keys()

    if kmer_q in keys_q:
        subs_q = kmer_dict_q[kmer_q]

        kmer_df = pd.DataFrame(index=np.arange(len(subs_q)))
        kmer_df['Peptide'] = subs_q
        return kmer_df
    else:
        raise KeyError('K-mer not in dataset. Please enter some other k-mer')

def similarity(str1, str2):
    """ Calculate Levenshtein distance similarity metrics between two strings

        Args:
            str1 (str): sequence 1 to compare
            str2 (str): sequence 2 to compare
        Returns:
            ratio (int): Levenshtein distance similarity ratio (as a percent)
            partial_ratio (int): Partial Levenshtein distance similarity ratio
                (as a percent). Ignores differences in length.
    """
    ratio = fuzz.ratio(str1.lower(), str2.lower())
    partial_ratio = fuzz.partial_ratio(str1.lower(), str2.lower())

    return ratio, partial_ratio

def similarity_matrix(subs_list, seqs_list, out_dir=None, close_plot=False):
    """ Calculate pairwise Levenshtein similarity between all substartes in
    subs_list and return similarity matrix.

        Args:
            subs_list (list, str): list of all substrate names of interest
            seqs_list (list, str): list of their corresponding sequences
            close_plot (bool): if True, close plots
            out_dir (str): directory path to save figures
        Returns:
            sim_m (pandas df): df of all subs_list x subs_list.
                Contains pairwise Levenshtein distance similarity ratio.
            sim_par_m (pandas df): df of all subs_list x subs_list,
                Contains pairwise Partial Levenshtein distance similarity ratio.
            cluster_grid_sim_m (sns clustermap): plot of hierarchically clustered
                heatmap of pairwise similarity.
            cluster_grid_sim_par_m (sns clustermap): plot of hierarchically
                clustered heatmap of pairwise partial similarity.
    """
    rows = subs_list
    cols = subs_list
    sim_m = pd.DataFrame(index=rows, columns=cols)
    sim_par_m = pd.DataFrame(index=rows, columns=cols)

    j = 0
    for j in np.arange(len(rows)):
        for i in np.arange(len(rows)):
            Str1 = seqs_list[j]
            Str2 = seqs_list[i]
            sim_m.iloc[j, i] = similarity(Str1, Str2)[0]
            sim_par_m.iloc[j, i] = similarity(Str1, Str2)[1]

        j = j + 1

    # Plot clustermap of similarity scores
    sim_m = sim_m.astype(float)
    sim_par_m = sim_par_m.astype(float)

    plt.figure()
    cluster_grid_sim_m = sns.clustermap(sim_m)
    plt.title('Levenshtein Similarity Ratio', fontsize=16)
    if out_dir is not None:
        plt.savefig(os.path.join(out_dir, 'sim_m.pdf'))
    if close_plot:
        plt.close()

    plt.figure()
    cluster_grid_sim_par_m = sns.clustermap(sim_par_m)
    plt.title('Partial Levenshtein Similarity Ratio', fontsize=16)
    if out_dir is not None:
        plt.savefig(os.path.join(out_dir, 'partial_sim_m.pdf'))
    if close_plot:
        plt.close()

    return sim_m, sim_par_m, cluster_grid_sim_m, cluster_grid_sim_par_m

def summarize_kmer(kmer_overlap_q, top_k, kmer_len, out_dir=None, close_plot=False):
    """ Summarize data of kmers overlapping a set of substrates, according to
    their frequency of occurence. Plot histogram of kmer distribution

        Args:
            kmer_overlap_q (dict): dictionary of kmers and their overlap with
                substrates, to be queried.
                keys: kmer sequences.
                values: substrates containg a given kmer.
            top_k (int): the number (k) of top kmers to display
            kmer_len (int): kmer length of interest to query
            out_dir (str): directory path for saving figure
            close_plot (bool): if True, close plots
        Returns:
            kmer_f_sorted (pandas df): df containing kmers sorted by frequency
                of occurrence
            kmer_f_sorted_filtered (pandas df): df containing the k top kmers
                sorted by their frequency
        """
    kmer_f_list = []
    for key in kmer_overlap_q.keys():
        kmer_f_list.append(len(kmer_overlap_q[key]))

    kmer_f = pd.DataFrame(index=kmer_overlap_q.keys())
    kmer_f['Frequency'] = kmer_f_list
    kmer_f_sorted = kmer_f.sort_values(by=['Frequency'], ascending=False)
    kmer_f_sorted_filtered = kmer_f_sorted.iloc[:top_k, :]

    # plot kmer histogram
    plt.figure()
    hist = kmer_f_sorted.hist(bins=np.max(np.max(kmer_f_sorted['Frequency'])))
    plt.xlabel('# substrates with kmer', fontsize=16)
    plt.ylabel('# kmers', fontsize=16)
    plt.title(str(kmer_len) + '-mer frequency distribution', fontsize=18)
    plt.savefig(os.path.join(out_dir, 'kmer_dist.pdf'))
    if close_plot:
        plt.close()

    return kmer_f_sorted, kmer_f_sorted_filtered
