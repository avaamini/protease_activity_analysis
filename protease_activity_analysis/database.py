""" Protease - substrate database."""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import os
import protease_activity_analysis as paa
from colorama import init, Fore, Back, Style

class SubstrateDatabase(object):

    def __init__(self, data_files, sequence_file, names_file=None, aa_dict_file=None):
        self.screens = {}
        self.file_list = data_files

        self.substrates = {}
        self.proteases = {}
        self.kmer_dict = {}
        self.kmer_overlap = {}

        self.screen_names = []

        unique_substrates = set()
        unique_proteases = set()

        for f in data_files:

            data, name, substrates, proteases = self.load_dataset(f)
            self.screen_names.append(name)
            self.screens[name] = data

            self.substrates[name] = substrates
            self.proteases[name] = proteases

            unique_substrates.update(substrates)
            unique_proteases.update(proteases)
        self.screen_substrates = list(unique_substrates)
        self.screen_proteases = list(unique_proteases)

        # load sequence information
        sequence_info = self.load_sequence_info(sequence_file)
        self.database = sequence_info
        self.unique_sequences = list(set(sequence_info['Sequence']))

        # Mapping of sequence names to alternative names/descriptors
        name_mapping = self.load_substrate_names(names_file)
        self.name_map = name_mapping

        # Mapping of AA one-letter code to RasMol color
        aa_d = self.load_aa_dict(aa_dict_file)
        self.aa_dict = aa_d

        # Summarize screen metadata - uncomment if we decide that this would always be worth running
        # self.summary_df = self.summarize_screen(names)

    def load_dataset(self, file_path, z_score=True):
        """ Load dataset from a csv file

        Args:
            file_path (str): path to the screening data file
            z_score (bool): whether or not to z-score data for standardization
        """

        data = pd.read_csv(file_path)
        file_name = os.path.basename(file_path).split('.csv')[0]

        data = data.groupby(data.columns[0]).mean().reset_index()
        substrates = list(data[data.columns[0]])
        proteases = list(data.iloc[:,1:].columns)

        #Correct for possible spaces
        proteases = [item.replace(' ', '') for item in proteases]
        data.columns = [data.columns[0]] + proteases  #Change names in df
        data = data.set_index(data.columns[0])

        if z_score:
            data = (data - data.mean()) / data.std(ddof=0)

        return data, file_name, substrates, proteases

    def load_substrate_names(self, names_file):
        """ Mapping of unified substrate names to alternative names or
        descriptors.

        Args:
            names_file (pkl): contains mapping of substrate names/nomenclature
                to lists of alternative names or supplementary descriptors

        Returns:
            substrate_dict (dictionary): names / descriptors mapping

        """
        f = open(names_file, 'rb')
        return pickle.load(f)

    def load_aa_dict(self, aa_dict_file):
        """ Mapping of AA one-letter code to RasMol color

        Args:
            aa_dict_file (pkl): contains mapping of AA one-letter code to RasMol color

        Returns:
            aa_dict (dictionary): AA / color mapping

        """
        f = open(aa_dict_file, 'rb')
        return pickle.load(f)

    def set_substrate_dict(self, substrate_dict):
        """ Set the database's substrate descriptor mapping
        """
        self.substrate_dict = substrate_dict

    def load_sequence_info(self, file_path):
        """ Load sequence information from csv file. Must have column 'Name'
        providing token names for the sequences, and column 'Sequence' providing
        sequences themselves. can have auxiliary names/descriptors.

        Args:
            file_path (str): path to the sequence information file
        Returns:
            seq_data (df): data frame of the sequence information
        """
        seq_data = pd.read_csv(file_path)
        seq_data = seq_data.set_index([seq_data.columns[0], 'Sequence', 'Composition'])

        # combine alternative names into list
        seq_data['Names'] = seq_data.values.tolist()
        seq_data.reset_index(inplace=True)
        seq_data.set_index(seq_data.columns[0])

        return seq_data

    # get_Screen_names broken --> doesnt return a lit but self.screen_names does
    def get_screen_names(self):
        """ Get the names of the screens in the database

        Returns:
            (list, str): screen names present in the database
        """
        return self.screen_names
        
    def get_screen_substrates(self, screen_name):
        """ Get all the substrates from a particular screen.

        Args:
            screen_name (str): the screen of interest.

        Returns:
            (list, str): substrate token names from screen of interest
        """
        return self.substrates[screen_name]

    def get_screen_proteases(self, screen_name):
        """ Get all the proteases from a particular screen.

        Args:
            screen_name (str): the screen of interest.

        Returns:
            (list, str): protease token names from screen of interest
        """
        return self.proteases[screen_name]

    def get_screen_data(self, screen_name):
        """ Get data from a particular screen.

        Args:
            screen_name (str): the screen of interest.

        Returns:
            (df): data from screen of interest
        """
        return self.screens[screen_name]

    def get_name_of_sequence(self, sequence):
        """ Get the substrate names for a sequence of interest.

        Args:
            sequence (str): sequence of interest.

        Returns:
            names (list, str): names for the sequence
        """
        sequence_data = self.database.loc[self.database['Sequence'] == sequence]

        if sequence_data.empty:
            raise ValueError("Substrate not found in database. Please try again.")

        name = sequence_data['Name']
        all_names = sequence_data['Names']
        alt_names = [x for x in all_names if x != 'nan']
        return name, all_names

    def get_sequence_of_name(self, name):
        """ Get the sequence for a substrate of interest.

        Args:
            name (str): name of substrate of interest.

        Returns:
            sequence (str): sequence for the name
        """
        name_data = self.database.loc[self.database['Name'] == name]

        if name_data.empty:
            name_data = self.database.loc[self.database['Name'] == self.get_unified_name(name)]
            if name_data.empty:
                raise ValueError("Substrate not found in database. Please try again.")

        seq = name_data['Sequence'].to_list()[0]

        return seq

    def get_kmer_dict(self, k):
        """ Get kmer_dict
        Args:
            k (int): k of interest
        Returns:
            (dict): kmer_dict
        """
        if k in self.kmer_dict.keys():
            return self.kmer_dict[k]
        else:
            print('No kmer_dict with key ' + str(k) + ' stored. Please use run_kmer_analysis() with said k prior to '
                                                      'calling this function')

    def get_kmer_overlap(self, k):
        """ Get kmer_overlap
        Args:
            k (int): k of interest
        Returns:
            (dict): kmer_overlap
        """
        if k in self.kmer_overlap.keys():
            return self.kmer_overlap[k]
        else:
            print('No kmer_overlap with key ' + str(k) + ' stored. Please use run_kmer_analysis() with said k prior to '
                                                         'calling this function')

    def search_protease(self, protease, out_dir=None, z_threshold=None):
        """ Return df of substrates and their cleavage by a given protease

        Args:
            protease (str): protease of interest
            out_dir (str): if specified, directory for writing data
            z_threshold (float): upper bound z-score cutoff for substrate return
        Returns:
            protease_cleavage_df (pandas df): zscores for protease against all
                substrates found in the screen datasets
        """
        protease_cleavage_dict = {}
        screens = self.proteases.keys()

        protease_found = False
        for screen in screens:
            if protease in self.proteases[screen]:
                print(f"{protease} found in {screen}")

                # get cleavage data for the protease found in particular screen
                protease_cleavage_dict[screen] = self.screens[screen][protease]
                protease_found = True

        if not protease_found:
            raise ValueError("Protease not found in datasets. Please try again.")

        protease_cleavage_df = pd.DataFrame.from_dict(protease_cleavage_dict)

        # Filter for values above the threshold
        if z_threshold is not None:
            protease_cleavage_df = protease_cleavage_df[protease_cleavage_df > z_threshold]
            protease_cleavage_df = protease_cleavage_df.dropna(how='all')

        if out_dir is not None:
            protease_cleavage_df.to_csv(
                os.path.join(out_dir, f"{protease}_{z_threshold}.csv")
            )

        return protease_cleavage_df

    def get_unified_name(self, substrate):
        """ Get the unified name for a substrate of interest

        Args:
            substrate (str): query of interest
        Returns:
            (str): unified nomenclature for the query
        """
        sub_info = self.database[self.database['Names'].apply(
            lambda x: substrate in x)
        ]
        if sub_info.empty:
            raise ValueError("Substrate not found in database. Please try again.")

        return sub_info['Name'].item()

    def search_substrate(self, substrate, out_dir=None, z_threshold=None):
        """ Return df of proteases and their cleavage of a given substrate

        Args:
            substrate (str): substrate of interest
            out_dir (str): if specified, directory for writing data
            z_threshold (float): upper bound z-score cutoff for substrate return
        Returns:
            protease_cleavage_df (pandas df): zscores for substrate against all
                proteases found in the screen datasets
        """
        screens = self.substrates.keys()

        def query_sub(sub, datasets):
            """ Get cleavage data for a substrate of interest"""
            sub_dict = {}
            for d in datasets:
                if sub in self.substrates[d]:
                    print(f"{sub} found in {d}")

                    # get cleavage data for the protease found in dataset
                    sub_dict[d] = self.screens[d].loc[sub]
            return sub_dict

        substrate_found = False
        substrate_cleavage_dict = query_sub(substrate, screens)

        if not substrate_cleavage_dict:
            print("Searching by alternative names...")
            sub_name = self.get_unified_name(substrate)

            if sub_name == '':
                raise ValueError(
                    "Substrate not found in datasets. Please try again."
                )
            print(f"Substrate found under unified name {sub_name}")
            substrate_cleavage_dict = query_sub(sub_name, screens)

        substrate_cleavage_df = pd.DataFrame.from_dict(substrate_cleavage_dict)

        # Filter for values above the threshold
        if z_threshold is not None:
            substrate_cleavage_df = substrate_cleavage_df[substrate_cleavage_df > z_threshold]
            substrate_cleavage_df = substrate_cleavage_df.dropna(how='all')

        if out_dir is not None:
            substrate_cleavage_df.to_csv(
                os.path.join(out_dir, f"{substrate}_{z_threshold}.csv")
            )

        return substrate_cleavage_df

    def search_sequence(self, sequence, out_dir=None, z_threshold=None):
        """ Return df of proteases and their cleavage of a given sequence

        Args:
            sequence (str): sequence of interest
            out_dir (str): if specified, directory for writing data
            z_threshold (float): upper bound z-score cutoff for substrate return
        Returns:
            protease_cleavage_df (pandas df): zscores for substrate against all
                proteases found in the screen datasets
        """
        seq_name, seq_alt_names = self.get_name_of_sequence(sequence)

        # search for the sequence in the database
        seq_cleavage_df = self.search_substrate(seq_name, out_dir, z_threshold)
        return seq_cleavage_df

    def get_top_hits(self, query, query_type, top_k, out_dir=None, z_threshold=None):
        """ Get the top hits from a cleavage dataset.

        Args:
            cleavage_df (df); data frame, indexed by proteases/substrates with
                columns corresponding to screens
            top_k (int): number of top hits to return
            out_dir (str): if specified, directory for writing data
            z_threshold (float): upper bound z-score cutoff

        Returns:

        """
        query_df = None
        if query_type == 'protease':
            query_df = self.search_protease(query, out_dir, z_threshold)
        elif query_type == 'substrate':
            query_df = self.search_substrate(query, out_dir, z_threshold)
        else:
            raise ValueError("Query must be either for a protease or a substrate")

        individual_dfs = []

        # Filter each dataset for the top K hits
        screens = query_df.columns
        for screen in screens:
            screen_data = pd.DataFrame(query_df[screen])
            screen_data.columns = ['Scores']
            screen_data.sort_values(by='Scores', ascending=False, inplace=True)

            top_k_screen = pd.DataFrame(screen_data[:top_k])
            top_k_screen.loc[:, 'Source'] = screen
            individual_dfs.append(top_k_screen)

            top_k_screen_names = list(top_k_screen.index)
            print(f"Top hits for {query} in {screen}:")
            if query_type == 'protease':
                for name in top_k_screen_names:
                    print(name + ' : ' + paa.substrate.color_seq(ex_sub=self.get_sequence_of_name(name),
                                                                 all_natural=
                                                                 self.database[self.database['Name'] == name][
                                                                     'Composition'].to_list() == ['Natural'],
                                                                 aa_dict=self.aa_dict))
                    print(Style.RESET_ALL)
            else:
                print(*top_k_screen_names, sep="\n")

        top_k_individual = pd.concat(individual_dfs)

        top_k_overall = pd.concat(individual_dfs)
        top_k_overall.sort_values(by='Scores', ascending=False, inplace=True)
        top_k_overall = top_k_overall[:top_k]

        top_k_print = zip(
            list(top_k_overall.index),
            list(top_k_overall['Source']),
            list(top_k_overall['Scores']))

        print(f"Top hits for {query} overall. Hit, Source, Score:")
        msg = "\n".join(
            "{}, {}, {:.2f}".format(hit, source, score) for hit, source, score in top_k_print
        )
        print(msg)

        if out_dir is not None:
            top_k_individual.to_csv(
                os.path.join(
                    out_dir,
                    f"{query}_{z_threshold}_{top_k}_individual.csv"
                )
            )

            top_k_overall.to_csv(
                os.path.join(
                    out_dir,
                    f"{query}_{z_threshold}_{top_k}_overall.csv"
                )
            )

        return top_k_individual, top_k_overall

    def plot_zscore_dist(self, df):
        """ Plot the distribution of zscores for a given matrix of cleavage data.

        Args:
            df (pandas df): data matrix with cleavage data of interest
        """
        return

    def summarize_screen(self, names):
        """ Summarize metadata from the different screens

        Args:
            names (list, str): screen names
        """
        col_summary_df = ['Screen', '# Peptides', '# Proteases', 'Metallo', 'Aspartic', 'Cysteine', 'Serine']
        summary_df = pd.DataFrame(columns=col_summary_df, index=np.arange(len(names)))

        for i in np.arange(len(names)):
            key = names[i]
            spec_temp = self.get_protease_class(names[i])
            summary_df.iloc[i, 3] = spec_temp[spec_temp['Class'] == 'Metallo'].shape[0]
            summary_df.iloc[i, 4] = spec_temp[spec_temp['Class'] == 'Aspartic'].shape[0]
            summary_df.iloc[i, 5] = spec_temp[spec_temp['Class'] == 'Cysteine'].shape[0]
            summary_df.iloc[i, 6] = spec_temp[spec_temp['Class'] == 'Serine'].shape[0]
            summary_df.loc[i, 0:3] = [key, len(self.substrates[key]), len(self.proteases[key])]

        ax1 = summary_df.plot.bar(x='Screen', y='# Peptides', rot=0, color='blue')
        ax1.set(title='# Peptides/Screen', xlabel='Screen', ylabel='# Peptides')

        labels = names
        metallo_f = summary_df['Metallo']
        aspartic_f = summary_df['Aspartic']
        cysteine_f = summary_df['Cysteine']
        serine_f = summary_df['Serine']

        fig, ax = plt.subplots()

        ax.bar(labels, cysteine_f, label='Cysteine', color='b')
        ax.bar(labels, aspartic_f, bottom=cysteine_f, label='Aspartic', color='k')
        ax.bar(labels, serine_f, bottom=cysteine_f + aspartic_f, label='Serine', color='orange')
        ax.bar(labels, metallo_f, bottom=cysteine_f + aspartic_f + serine_f, label='Metallo', color='g')

        ax.set_ylabel('Frequency')
        ax.set_title('#Proteases/screen by class')
        ax.legend()

        plt.show()
        return summary_df

    def run_kmer_analysis(self, k_mer_list):
        """ First filters for natural substrates onlt and then generates dictionaries cantaining all kmers of lengths of interest and finds peptides containgn overlapping kmers in the dataset
            Args:
                k_mer_list (list, int): kmer lengths of interest to query
            Returns:

        """
        # Filter for natural substrates only
        natural = self.database[self.database['Composition'] == 'Natural']
        # print(natural)

        # Run generate_kmer function for all peptides in PAA for all kmer lengths of interest
        subs = natural.iloc[:, 0].to_list()  # First column to be most generalizable must contain substrate names
        seqs = natural['Sequence'].to_list()

        for k in k_mer_list:
            self.kmer_dict[k] = paa.substrate.generate_kmers(subs, seqs, k)

        # Populate kmer_overlap dictionaries
        for k in k_mer_list:
            print('For k=', k)
            self.kmer_overlap[k] = paa.substrate.find_overlap(self.kmer_dict[k])

    def search_kmer(self, kmer_q, all_natural, aa_dict):
        """ Returns substrates in database containing kmer of interest
            Args:
                kmer_q (str): kmer to query
                all_natural (bool): if True color-code, if False return regular sequence
                aa_dict (dictionary): aa dictionary with color scheme to use for each one-letter AA
            Returns:
                df (pandas df): dataframe containing AA
        """
        kmer_len = len(kmer_q)

        #     f = open('data/screens/PAA/kmer_analyses/kmer_'+str(kmer_len)+'_paa.pickle', 'rb')
        #     kmer_dict_q = pickle.load(f)

        if kmer_len in self.kmer_overlap.keys():
            kmer_dict_q = self.kmer_overlap[kmer_len]

            keys_q = kmer_dict_q.keys()

            if kmer_q in keys_q:
                subs_q = kmer_dict_q[kmer_q]

                seqs_q = []
                for seq in subs_q:
                    seqs_q.append(self.get_sequence_of_name(seq))
                    print(seq + ':' + paa.substrate.color_seq(self.get_sequence_of_name(seq), all_natural, aa_dict))
                    print(Style.RESET_ALL)
                df = pd.DataFrame(index=np.arange(len(subs_q)))
                df['Peptide'] = subs_q
                df['Sequence'] = seqs_q

            else:
                print('K-mer not in dataset, please enter some other k-mer')
                df = None
        else:
            print('No information for kmer of length ' + str(
                kmer_len) + ' stored. Please use run_kmer_analysis() with said k prior to '
                            'calling this function')
            df = None

        return df

    def find_similar_substrates(self, seq, all_natural, metric, top_k):
        """ Compute similarity between the sequence of interest and substrates in teh database
            Args:
                seq (str): AA sequence of interest
                all_natural (bool): if True color-code, if False return regular sequence
                metric (str): similarity metric to sort by. 2 options: 'Similarity Ratio' or 'Partial Similarity Ratio'
                top_k (int): top_k most similar sequences to print
            Returns:
                sim_m_sorted (pandas df): df of all sequences in the database and their similarity to the sequence of interest
        """
        sim_m = self.database.copy()
        sim_m = sim_m.iloc[:, 0:3]
        sim_m['Similarity Ratio'] = sim_m.apply(lambda row: paa.substrate.similarity(seq, row['Sequence'])[0], axis=1)
        sim_m['Partial Similarity Ratio'] = sim_m.apply(lambda row: paa.substrate.similarity(seq, row['Sequence'])[1],
                                                        axis=1)
        sim_m_sorted = sim_m.sort_values(by=[metric], ascending=False)

        print('Queried seq:')
        print(paa.substrate.color_seq(seq, all_natural, self.aa_dict))
        print(Style.RESET_ALL)

        top_k = sim_m_sorted.iloc[:top_k, :]
        for i in np.arange(top_k.shape[0]):
            print(top_k['Name'].iloc[i] + ':' + paa.substrate.color_seq(top_k['Sequence'].iloc[i],
                                                          top_k['Composition'].iloc[i] == 'Natural', self.aa_dict))
            print(Style.RESET_ALL)

        return sim_m_sorted, top_k

    def get_similarity_matrix(self):
        """ Calculate pairwise similarity between all substartes in subs_list and return similarity matrix
            Args:
                subs_list (list, str): list containing all names of substrates of interest
                seqs_list (list, str): list containing their corresponding sequences
            Returns:
                sim_m (pandas df): df of all subs_list x subs_list containig pairwise Levenshtein distance similarity ratio (Ratio)
                sim_par_m (pandas df): df of all subs_list x subs_list containig pairwise Partial Levenshtein distance similarity ratio (Ratio)
        """
        subs_list = self.database.iloc[:, 0].to_list()
        seqs_list = self.database['Sequence'].to_list()
        sim_m = pd.DataFrame(index=subs_list, columns=subs_list)
        sim_par_m = pd.DataFrame(index=subs_list, columns=subs_list)

        j = 0
        for j in np.arange(len(subs_list)):
            for i in np.arange(len(subs_list)):
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

        return sim_m, sim_par_m

    def summarize_kmer(self, kmer_len, top_k):
        """ Summarize kmer_overlap data
            Args:
                kmer_len (dictionary): kmer_length to summarize
                top_k (int): top_k kmers to display
            Returns:
                kmer_f_sorted (pandas df): dataframe containing sorted kmers by their frequency
                kmer_f_        sorted_filtered (pandas df): dataframe containing top_k sorted kmers by their frequency
        """
        kmer_overlap_q = self.get_kmer_overlap(kmer_len)
        kmer_f_list = []
        for key in kmer_overlap_q.keys():
            kmer_f_list.append(len(kmer_overlap_q[key]))

        kmer_f = pd.DataFrame(index=kmer_overlap_q.keys())
        kmer_f['Frequency'] = kmer_f_list
        kmer_f_sorted = kmer_f.sort_values(by=['Frequency'], ascending=False)
        kmer_f_sorted_filtered = kmer_f_sorted.iloc[:top_k, :]

        hist = kmer_f_sorted.hist(bins=np.max(np.max(kmer_f_sorted['Frequency'])))

        return kmer_f_sorted, kmer_f_sorted_filtered

    def get_protease_class(self, screen_name):
        """ Get class of proteases in a screen
        Args:
             screen_name (str): screen name to look up protease class of
        Returns:
            protease_class_dict (dictionary): dictionary with protease class of each prtoease in the screen
        """
        prot = self.get_screen_proteases(screen_name)
        protease_class_dict = pd.DataFrame(data={'Protease': prot}, index=np.arange(len(prot)))
        protease_class_dict['Class'] = protease_class_dict.apply(
            lambda row: paa.protease.classify_protease(row['Protease']), axis=1)

        return protease_class_dict
