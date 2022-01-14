""" Sequence-related functions"""
import pandas as pd
import scipy as sp
from scipy import stats
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from scipy.stats import zscore
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import scipy.stats as ss
import csv
import pickle
from colorama import init, Fore, Back, Style
from fuzzywuzzy import fuzz
import protease_activity_analysis as paa


def color_seq(ex_sub, all_natural, aa_dict):
    """ Returns AA sequence color-coded by RasMol color scheme

        Args:
            ex_sub (str): AA sequence to color code
            all_natural (bool): if True color-code, if False return regular sequence
            aa_dict (dictionary): aa dictionary with color scheme to use for each one-letter AA
        Returns:
            color_sub (str): color-coded AA sequence
    """
    len_ex_sub = len(ex_sub)

    if all_natural:
        if len_ex_sub == 1:
            color_sub = aa_dict[ex_sub[0]]
        elif len_ex_sub == 2:
            color_sub = aa_dict[ex_sub[0]] + aa_dict[ex_sub[1]]
        elif len_ex_sub == 3:
            color_sub = aa_dict[ex_sub[0]] + aa_dict[ex_sub[1]] + aa_dict[ex_sub[2]]
        elif len_ex_sub == 4:
            color_sub = aa_dict[ex_sub[0]] + aa_dict[ex_sub[1]] + aa_dict[ex_sub[2]] + aa_dict[ex_sub[3]]
        elif len_ex_sub == 5:
            color_sub = aa_dict[ex_sub[0]] + aa_dict[ex_sub[1]] + aa_dict[ex_sub[2]] + aa_dict[ex_sub[3]] + aa_dict[
                ex_sub[4]]
        elif len_ex_sub == 6:
            color_sub = aa_dict[ex_sub[0]] + aa_dict[ex_sub[1]] + aa_dict[ex_sub[2]] + aa_dict[ex_sub[3]] + aa_dict[
                ex_sub[4]] + aa_dict[ex_sub[5]]
        elif len_ex_sub == 7:
            color_sub = aa_dict[ex_sub[0]] + aa_dict[ex_sub[1]] + aa_dict[ex_sub[2]] + aa_dict[ex_sub[3]] + aa_dict[
                ex_sub[4]] + aa_dict[ex_sub[5]] + aa_dict[ex_sub[6]]
        elif len_ex_sub == 8:
            color_sub = aa_dict[ex_sub[0]] + aa_dict[ex_sub[1]] + aa_dict[ex_sub[2]] + aa_dict[ex_sub[3]] + aa_dict[
                ex_sub[4]] + aa_dict[ex_sub[5]] + aa_dict[ex_sub[6]] + aa_dict[ex_sub[7]]
        elif len_ex_sub == 9:
            color_sub = aa_dict[ex_sub[0]] + aa_dict[ex_sub[1]] + aa_dict[ex_sub[2]] + aa_dict[ex_sub[3]] + aa_dict[
                ex_sub[4]] + aa_dict[ex_sub[5]] + aa_dict[ex_sub[6]] + aa_dict[ex_sub[7]] + aa_dict[ex_sub[8]]
        elif len_ex_sub == 10:
            color_sub = aa_dict[ex_sub[0]] + aa_dict[ex_sub[1]] + aa_dict[ex_sub[2]] + aa_dict[ex_sub[3]] + aa_dict[
                ex_sub[4]] + aa_dict[ex_sub[5]] + aa_dict[ex_sub[6]] + aa_dict[ex_sub[7]] + aa_dict[ex_sub[8]] + \
                        aa_dict[ex_sub[9]]
        elif len_ex_sub == 11:
            color_sub = aa_dict[ex_sub[0]] + aa_dict[ex_sub[1]] + aa_dict[ex_sub[2]] + aa_dict[ex_sub[3]] + aa_dict[
                ex_sub[4]] + aa_dict[ex_sub[5]] + aa_dict[ex_sub[6]] + aa_dict[ex_sub[7]] + aa_dict[ex_sub[8]] + \
                        aa_dict[ex_sub[9]] + aa_dict[ex_sub[10]]
        elif len_ex_sub == 12:
            color_sub = aa_dict[ex_sub[0]] + aa_dict[ex_sub[1]] + aa_dict[ex_sub[2]] + aa_dict[ex_sub[3]] + aa_dict[
                ex_sub[4]] + aa_dict[ex_sub[5]] + aa_dict[ex_sub[6]] + aa_dict[ex_sub[7]] + aa_dict[ex_sub[8]] + \
                        aa_dict[ex_sub[9]] + aa_dict[ex_sub[10]] + aa_dict[ex_sub[11]]
        elif len_ex_sub == 13:
            color_sub = aa_dict[ex_sub[0]] + aa_dict[ex_sub[1]] + aa_dict[ex_sub[2]] + aa_dict[ex_sub[3]] + aa_dict[
                ex_sub[4]] + aa_dict[ex_sub[5]] + aa_dict[ex_sub[6]] + aa_dict[ex_sub[7]] + aa_dict[ex_sub[8]] + \
                        aa_dict[ex_sub[9]] + aa_dict[ex_sub[10]] + aa_dict[ex_sub[11]] + aa_dict[ex_sub[12]]
        elif len_ex_sub == 14:
            color_sub = aa_dict[ex_sub[0]] + aa_dict[ex_sub[1]] + aa_dict[ex_sub[2]] + aa_dict[ex_sub[3]] + aa_dict[
                ex_sub[4]] + aa_dict[ex_sub[5]] + aa_dict[ex_sub[6]] + aa_dict[ex_sub[7]] + aa_dict[ex_sub[8]] + \
                        aa_dict[ex_sub[9]] + aa_dict[ex_sub[10]] + aa_dict[ex_sub[11]] + aa_dict[ex_sub[12]] + aa_dict[
                            ex_sub[13]]
        else:
            print('Substrate length out of range')
            print(Style.RESET_ALL)
            # print(ex_sub)
            color_sub = ex_sub
    else:
        print(Style.RESET_ALL)
        color_sub = ex_sub

    return color_sub


def generate_kmers(sub_list, seq_list, k):
    """ Returns dictionary containing kmers of length k in each substrate

        Args:
            sub_list (list, str): list of all substrates
            seq_list (list, str): list of sequences
            k (int): kmer length
        Returns:
            kmer_dict (dict): dictionary where keys are substrates and values are kmers of length k in a given substrate
    """
    kmer_dict = {}  # create empy dictionary, keys will be the name of the peptide and the values will be the kmers
    for i in np.arange(len(sub_list)):
        temp_seq = seq_list[i]
        # print(temp_seq)
        # print(len(temp_seq))
        n = len(temp_seq) - k + 1
        # print(n)
        temp_kmer = []
        for j in np.arange(n): temp_kmer.append(temp_seq[j:j + k])
        kmer_dict[sub_list[i]] = temp_kmer

    return kmer_dict


def find_overlap(kmer_dict):
    """ Returns dictionary where keys are kmers and values are substrates containing each kmer
        Args:
            kmer_dict (dict): dictionary where keys are substrates and values are kmers of length k in a given substrate
        Returns:
            overlap_dict (dict): dictionary where keys are kmers and values are substrates containg a given kmer
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
    """ Search kmer of length k in a given kmer_dict_q
        Args:
            kmer_q (str): kmer to query
            kmer_dict_q (dictionary): dictionary containing kmer breakdown. Must have kmers with same length as kmer_q
             being queried
        Returns:
            df (pandas df): dataframe containing AA
    """
    keys_q = kmer_dict_q.keys()

    if kmer_q in keys_q:
        subs_q = kmer_dict_q[kmer_q]

        df = pd.DataFrame(index=np.arange(len(subs_q)))
        df['Peptide'] = subs_q

    else:
        print('K-mer not in dataset, please enter some other k-mer')
        df = None

    return df


def similarity(str1, str2):
    """ Calculate similarity metrics between two strings
        Args:
            str1 (str): sequence 1 to compare
            str2 (str): sequence 2 to compare
        Returns:
            ratio (int): Levenshtein distance similarity ratio (Ratio) as a percent
            partial_ratio (int): Partial Levenshtein distance similarity ratio (Ratio) as a percent
    """
    # Calculate Levenshtein distance similarity ratio (Ratio) as well as a Partial Levenshtein distance similarity ratio that ignores differences in length
    ratio = fuzz.ratio(str1.lower(), str2.lower())
    partial_ratio = fuzz.partial_ratio(str1.lower(), str2.lower())

    return ratio, partial_ratio


def similarity_matrix(subs_list, seqs_list):
    """ Calculate pairwise similarity between all substartes in subs_list and return similarity matrix
        Args:
            subs_list (list, str): list containing all names of substrates of interest
            seqs_list (list, str): list containing their corresponding sequences
        Returns:
            sim_m (pandas df): df of all subs_list x subs_list containig pairwise Levenshtein distance similarity ratio (Ratio)
            sim_par_m (pandas df): df of all subs_list x subs_list containig pairwise Partial Levenshtein distance similarity ratio (Ratio)
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
            sim_m.iloc[j, i] = paa.substrate.similarity(Str1, Str2)[0]
            sim_par_m.iloc[j, i] = paa.substrate.similarity(Str1, Str2)[1]

        j = j + 1

    # Plot clustermap of similarity scores
    sim_m = sim_m.astype(float)
    sim_par_m = sim_par_m.astype(float)

    plt.figure()
    cluster_grid_sim_m = sns.clustermap(sim_m)
    plt.title('Levenschtein Similarity Ratio', fontsize=16)

    plt.figure()
    cluster_grid_sim_par_m = sns.clustermap(sim_par_m)
    plt.title('Partial Levenschtein Similarity Ratio', fontsize=16)
    return sim_m, sim_par_m, cluster_grid_sim_m, cluster_grid_sim_par_m


def summarize_kmer(kmer_overlap_q, top_k):
    """ Summarize kmer_overlap data
        Args:
            kmer_overlap_q (dictionary): kmer_overlap dictionary to query
            top_k (int): top_k kmers to display
        Returns:
            kmer_f_sorted (pandas df): dataframe containing sorted kmers by their frequency
            kmer_f_        sorted_filtered (pandas df): dataframe containing top_k sorted kmers by their frequency
        """
    kmer_f_list = []
    for key in kmer_overlap_q.keys():
        kmer_f_list.append(len(kmer_overlap_q[key]))

    kmer_f = pd.DataFrame(index=kmer_overlap_q.keys())
    kmer_f['Frequency'] = kmer_f_list
    kmer_f_sorted = kmer_f.sort_values(by=['Frequency'], ascending=False)
    kmer_f_sorted_filtered = kmer_f_sorted.iloc[:top_k, :]

    hist = kmer_f_sorted.hist(bins=np.max(np.max(kmer_f_sorted['Frequency'])))

    return kmer_f_sorted, kmer_f_sorted_filtered






