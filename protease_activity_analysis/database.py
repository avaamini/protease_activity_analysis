""" Protease - substrate database for query and analysis."""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import os
import protease_activity_analysis as paa
from colorama import init, Fore, Back, Style
from adjustText import adjust_text

class SubstrateDatabase(object):

    def __init__(self, data_files, sequence_file, names_file=None,
        aa_dict_file=None):
        self.screens = {}
        self.raw_screens = {}
        self.raw_limits = {}
        self.file_list = data_files

        self.substrates = {}
        self.proteases = {}
        self.kmer_dict = {}
        self.kmer_overlap = {}

        self.screen_names = []

        unique_substrates = set()
        unique_proteases = set()

        for f in data_files:

            data_zscored, data, name, substrates, proteases = self.load_dataset(f)
            self.screen_names.append(name)
            self.screens[name] = data_zscored
            self.raw_screens[name] = data
            self.raw_limits[name] = [np.nanmin(data.values),
                np.nanmax(data.values)]

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

    def load_dataset(self, file_path):
        """ Load protease-substrate dataset from a csv file. Prepare dataframes
        of the dataset, including raw and zscored data.

        Args:
            file_path (str): path to the screening data file
        Returns:
            data_zscored (df): standardized cleavage scores for data in input
                file
            data (df): raw cleavage values for data in input file
            file_name (str): base filename for the input data
            substrates (list str): names of substrates found in the dataset
            proteases (list str): names of proteases found in the dataset
        """
        # load data into dataframe
        data = pd.read_csv(file_path)
        file_name = os.path.basename(file_path).split('.csv')[0]

        data = data.groupby(data.columns[0]).mean().reset_index()
        substrates = list(data[data.columns[0]])
        proteases = list(data.iloc[:,1:].columns)

        # Correct for possible spaces
        proteases = [item.replace(' ', '') for item in proteases]
        data.columns = [data.columns[0]] + proteases  # Change names in df
        data = data.set_index(data.columns[0])

        data_zscored = (data - data.mean()) / data.std(ddof=0)

        return data_zscored, data, file_name, substrates, proteases

    def load_substrate_names(self, names_file):
        """ Mapping of unified substrate names to alternative names or
        descriptors.

        Args:
            names_file (pkl): contains mapping of substrate names/nomenclature
                to lists of alternative names or supplementary descriptors

        Returns:
            substrate_name_dict (dict): dictionary mapping substrate names to
                alternative names or supplementary descriptors.
                keys (str): substrate name
                values (list str): alternative names/descriptors
        """
        f = open(names_file, 'rb')
        substrate_name_dict = pickle.load(f)
        return substrate_name_dict

    def load_aa_dict(self, aa_dict_file):
        """ Mapping of AA one-letter code to RasMol color

        Args:
            aa_dict_file (pkl): mapping of AA code to RasMol color

        Returns:
            aa_dict (dict): dictionary mapping AA code to RasMol color

        """
        f = open(aa_dict_file, 'rb')
        aa_dict = pickle.load(f)
        return aa_dict

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
        seq_data = seq_data.set_index(
            [seq_data.columns[0], 'Sequence', 'Composition']
        )

        # combine alternative names into list
        seq_data['Names'] = seq_data.values.tolist()
        seq_data.reset_index(inplace=True)
        seq_data.set_index(seq_data.columns[0])

        return seq_data

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
            raise ValueError("Substrate not found in database. Try again.")

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
            name_data = self.database.loc[self.database['Name'] == \
                self.get_unified_name(name)]
            if name_data.empty:
                raise ValueError("Substrate not found in database. Try again.")

        seq = name_data['Sequence'].to_list()[0]

        return seq

    def get_kmer_dict(self, k):
        """ Get the dictionary of kmers of length k, mapping substrates to kmers.

        Args:
            k (int): kmer length of interest
        Returns:
            (dict): dictionary where keys are substrate names.
                values are kmers of length k in a given substrate.
        Raises:
            KeyError: if kmer length k is not present in the set of k_mer
                dictionaries stored in the database.
        """
        if k in self.kmer_dict.keys():
            return self.kmer_dict[k]
        else:
            raise KeyError(f'No kmer_dict with kmer length {k} stored. Use \
                run_kmer_analysis() with said k prior to calling this function.')

    def get_kmer_overlap(self, k):
        """ Get dictionary of all kmer sequences of length k, mapped to substrate
        sequencing containing that kmer.

        Args:
            k (int): kmer length of interest
        Returns:
            (dict): dictionary where keys are kmer sequences of length k.
                values are substrates containg the kmer of length k.
        """
        if k in self.kmer_overlap.keys():
            return self.kmer_overlap[k]
        else:
            raise KeyError(f'No kmer overlap with kmer length {k} stored. Use \
                run_kmer_analysis() with said k prior to calling this function')

    def search_protease(self, protease, out_dir=None, z_threshold=None, plot=True):
        """ Return df of substrates and their cleavage by a given protease

        Args:
            protease (str): protease of interest
            out_dir (str): if specified, directory for writing data
            z_threshold (float): lower bound (exclusive) z-score cutoff.
                all entries with z-score > z_threshold will be returned
            plot (bool): if True, plot summary histogram
        Returns:
            protease_cleavage_df (pandas df): zscores for protease against all
                substrates found in the screen datasets
        """
        protease_cleavage_dict = {}
        screens = self.proteases.keys()
        all_zscores = []

        protease_found = False
        for screen in screens:
            if protease in self.proteases[screen]:
                print(f"{protease} found in {screen}")

                # get cleavage data for the protease found in particular screen
                protease_cleavage_dict[screen] = self.screens[screen][protease]
                all_zscores.append(protease_cleavage_dict[screen].values)
                protease_found = True

        if not protease_found:
            raise ValueError("Protease not found in datasets. Try again.")

        # Plot histogram of screening data
        if plot:
            paa.vis.hist(
                all_zscores,
                screens,
                'z_scores',
                'Frequency',
                'Z-score distributions',
                protease,
                out_dir
            )

        protease_cleavage_df = pd.DataFrame.from_dict(protease_cleavage_dict)

        # Filter for values above the threshold
        if z_threshold is not None:
            protease_cleavage_df = protease_cleavage_df[ \
                protease_cleavage_df > z_threshold]
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
            raise ValueError("Substrate not found in database. Try again.")

        return sub_info['Name'].item()

    def search_substrate(self, substrate, out_dir=None, z_threshold=None):
        """ Return df of proteases and their cleavage of a given substrate.
        Write df to a csv file.

        Args:
            substrate (str): substrate of interest
            out_dir (str): if specified, directory for writing data
            z_threshold (float): lower bound (exclusive) z-score cutoff.
                all entries with z-score > z_threshold will be returned
        Returns:
            substrate_cleavage_df (pandas df): zscores for substrate against all
                proteases found in the screen datasets
            sub_name (str): name of substrate as it was found in the database.
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

        # search the database by the provided name
        substrate_cleavage_dict = query_sub(substrate, screens)

        if not substrate_cleavage_dict: # input substrate name not found
            print("Searching by alternative names...")
            sub_name = self.get_unified_name(substrate)

            if sub_name == '': # no matches under unified or alternative names
                raise ValueError(
                    "Substrate not found in datasets. Please try again."
                )

            substrate_cleavage_dict = query_sub(sub_name, screens)
            if substrate_cleavage_dict: # substrate was found under unified name
                print(f"Substrate found under unified name {sub_name}")
        else: # substrate found under inputted name
            sub_name = substrate

        substrate_cleavage_df = pd.DataFrame.from_dict(substrate_cleavage_dict)

        # Filter for values above the threshold
        if z_threshold is not None:
            substrate_cleavage_df = substrate_cleavage_df[ \
                substrate_cleavage_df > z_threshold]
            substrate_cleavage_df = substrate_cleavage_df.dropna(how='all')

        if out_dir is not None:
            substrate_cleavage_df.to_csv(
                os.path.join(out_dir, f"{substrate}_{z_threshold}.csv")
            )

        return substrate_cleavage_df, sub_name

    def search_sequence(self, sequence, out_dir=None, z_threshold=None):
        """ Return df of proteases and their cleavage of a given sequence

        Args:
            sequence (str): sequence of interest
            out_dir (str): if specified, directory for writing data
            z_threshold (float): lower bound (exclusive) z-score cutoff.
                all entries with z-score > z_threshold will be returned
        Returns:
            seq_cleavage_df (pandas df): zscores for cleavage of sequence by
                proteases found in the screen datasets
        """
        seq_name, seq_alt_names = self.get_name_of_sequence(sequence)

        # search for the sequence in the database
        seq_cleavage_df = self.search_substrate(seq_name, out_dir, z_threshold)
        return seq_cleavage_df

    def get_top_hits(self, query, query_type, k=5, out_dir=None, z_threshold=None):
        """ Get the top $k$ hits from a cleavage dataset, and write the data to
        file. Can query by a protease, substrate name, or substrate sequence.

        Args:
            query (str): name of query of interest. Either a protease, substrate
                name, or substrate sequence.
            query_type (str): type of query: 'protease', 'substrate', 'sequence'
            k (int): number of top hits to return
            out_dir (str): if specified, directory for writing data
            z_threshold (float): lower bound (exclusive) z-score cutoff.
                all entries with z-score > z_threshold will be returned

        Returns:
            top_k_individual (df): top k hits for each of the datasets
                present in the database.
            top_k_overall (df): top k hits overall, over the database as a
                collective.
        """
        query_df = None
        if query_type == 'protease':
            query_df = self.search_protease(query, out_dir, z_threshold)
        elif query_type == 'substrate':
            query_df = self.search_substrate(query, out_dir, z_threshold)
        elif query_type == 'sequence':
            query_df = self.search_sequence(query, out_dir, z_threshold)
        else:
            raise ValueError("Query must be either a protease or a substrate. \
                Substrates can be specified by name or sequence.")

        individual_dfs = []

        # Filter each dataset for the top K hits
        screens = query_df.columns
        for screen in screens:
            screen_data = pd.DataFrame(query_df[screen])
            screen_data.columns = ['Scores']
            screen_data.sort_values(by='Scores', ascending=False, inplace=True)

            top_k_screen = pd.DataFrame(screen_data[:k])
            top_k_screen.loc[:, 'Source'] = screen
            individual_dfs.append(top_k_screen)

            top_k_screen_names = list(top_k_screen.index)
            print(f"Top hits for {query} in {screen}:")
            if query_type == 'protease':
                for name in top_k_screen_names:
                    # get AA sequence and assess if all natural AA
                    sequence_of_name = self.get_sequence_of_name(name)
                    seq_all_natural = self.database[ \
                        self.database['Name'] == name][ \
                            'Composition'].to_list() == ['Natural']

                    # color-code amino acids
                    colored_sequence = paa.substrate.color_seq(
                        sequence_of_name,
                        seq_all_natural,
                        self.aa_dict
                    )
                    # print the substrate sequences
                    print(f"{name}: {sequence_of_name} - {colored_sequence}")
                    print(Style.RESET_ALL)
            else: # print the top proteases
                print(*top_k_screen_names, sep="\n")

        # top k in each screen individually
        top_k_individual = pd.concat(individual_dfs)
        # top k overall across all screens in database

        top_k_overall = pd.concat(individual_dfs)
        top_k_overall.sort_values(by='Scores', ascending=False, inplace=True)
        top_k_overall = top_k_overall[:top_k]

        if query_type == 'protease':
            top_k_individual = top_k_individual.reset_index()
            top_k_individual['Sequence'] = top_k_individual.apply(
                lambda row: self.get_sequence_of_name(row['index']), axis=1
            )
            top_k_individual = top_k_individual.set_index('index')

            top_k_overall = top_k_overall.reset_index()
            top_k_overall['Sequence'] = top_k_overall.apply(
                lambda row: self.get_sequence_of_name(row['index']), axis=1
            )
            top_k_overall = top_k_overall.set_index('index')

        top_k_print = zip(
            list(top_k_overall.index),
            list(top_k_overall['Source']),
            list(top_k_overall['Scores'])
        )

        print(f"Top hits for {query} overall. Hit, Source, Score:")
        msg = "\n".join(
            "{}, {}, {:.2f}".format(hit, source, score) \
                for hit, source, score in top_k_print
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

    def summarize_screen(self, names, out_dir=None, close_plot=True):
        """ Summarize metadata from the different screens. Summary plots
        of the metadata.

        Args:
            names (list, str): screen names
            out_dir (str): if specified, directory for writing data
            close_plot (bool): if True, close plots
        Returns:
            summary_df (df): summary metadata across all the screens in the
                database.
        """
        col_summary_df = [
            'Screen',
            '# Peptides',
            '# Proteases',
            'Metallo',
            'Aspartic',
            'Cysteine',
            'Serine'
        ]
        summary_df = pd.DataFrame(
            columns=col_summary_df, index=np.arange(len(names))
        )

        for i in np.arange(len(names)):
            key = names[i]
            spec_temp = self.get_protease_class(names[i])
            summary_df.iloc[i, 3] = spec_temp[ \
                spec_temp['Class'] == 'Metallo'].shape[0]
            summary_df.iloc[i, 4] = spec_temp[ \
                spec_temp['Class'] == 'Aspartic'].shape[0]
            summary_df.iloc[i, 5] = spec_temp[ \
                spec_temp['Class'] == 'Cysteine'].shape[0]
            summary_df.iloc[i, 6] = spec_temp[ \
                spec_temp['Class'] == 'Serine'].shape[0]
            summary_df.loc[i, 0:3] = [ \
                key, len(self.substrates[key]), len(self.proteases[key])]

        # plotting summary metrics for substrates
        ax1 = summary_df.plot.bar(x='Screen', y='# Peptides', rot=0, color='b')
        ax1.set_title('# Peptides/Screen', fontsize=20)
        ax1.set_xlabel('Screen', fontsize=18)
        ax1.set_ylabel('# Peptides', fontsize=18)
        plt.xticks(rotation=45, ha='right', fontsize=15)
        ax1.legend(prop={'size': 11})
        file_path = os.path.join(out_dir, 'summary_screens.pdf')
        ax1.figure.savefig(file_path)
        plt.show()
        if close_plot:
            plt.close()

        # plotting summary metrics for proteases
        labels = names
        metallo_f = summary_df['Metallo']
        aspartic_f = summary_df['Aspartic']
        cysteine_f = summary_df['Cysteine']
        serine_f = summary_df['Serine']

        fig, ax = plt.subplots()

        ax.bar(labels, cysteine_f,
            label='Cysteine', color='b')
        ax.bar(labels, aspartic_f, bottom=cysteine_f,
            label='Aspartic', color='k')
        ax.bar(labels, serine_f, bottom=cysteine_f + aspartic_f,
            label='Serine', color='orange')
        ax.bar(labels, metallo_f, bottom=cysteine_f + aspartic_f + serine_f,
            label='Metallo', color='g')

        ax.set_ylabel('Frequency', fontsize=18)
        ax.set_title('#Proteases/screen by class', fontsize=20)
        plt.xticks(rotation=45, ha='right', fontsize=15)
        ax.legend(prop={'size': 11})
        file_path = os.path.join(out_dir, 'summary_protease.pdf')
        ax.figure.savefig(file_path)
        plt.show()

        if close_plot:
            plt.close()

        plt.show()
        return summary_df

    def run_kmer_analysis(self, kmer_lengths):
        """ Perform kmer analysis over the database.
        First filter for substrates with natural amino acids.
        Generate dictionaries containing all kmers of lengths of interest.
        Search and find peptides containing the kmers present in the database.

        Set dictionaries as database attributes.

            Args:
                kmer_lengths (list, int): kmer lengths of interest to query

        """
        # Filter for natural substrates only
        natural = self.database[self.database['Composition'] == 'Natural']

        # Run generate_kmer function for all peptides for all kmer lengths
        subs = natural.iloc[:, 0].to_list()  # First column: substrate names
        seqs = natural['Sequence'].to_list()

        # map substrates to kmers, and then populate overlap dictionaries
        #   mapping kmers to substrates containing those kmers
        for k in kmer_lengths:
            print(f'Generating kmer dictionary for length {k}')
            self.kmer_dict[k] = paa.substrate.generate_kmers(subs, seqs, k)
            print(f'Generating kmer overlap dictionary for length {k}')
            self.kmer_overlap[k] = paa.substrate.find_overlapping_kmers(
                self.kmer_dict[k])
        print("Completed kmer analysis for all lengths provided.")

    def search_kmer(self, kmer_q, all_natural, aa_dict):
        """ Returns substrates in database containing query kmer of interest

            Args:
                kmer_q (str): kmer sequence to query
                all_natural (bool): if True color-code, if False print uncolored
                aa_dict (dictionary): color scheme to use for each one-letter AA
            Returns:
                substrates_with_kmer (pandas df): dataframe containing substrate
            Raises:
                KeyError: when query kmer not found in database
        """
        kmer_len = len(kmer_q)

        # find the kmer overlap dictionary for the length of the query kmer,
        #   and then extract the substrate sequences containing that kmer
        if kmer_len in self.kmer_overlap.keys():
            kmer_dict_q = self.kmer_overlap[kmer_len]

            keys_q = kmer_dict_q.keys()

            if kmer_q in keys_q: # found query kmer
                subs_q = kmer_dict_q[kmer_q]

                seqs_q = []
                for sub in subs_q:
                    seq_i = self.get_sequence_of_name(sub)
                    seqs_q.append(seq_i)
                    # print out information about the substrates found
                    colored_sequence = paa.substrate.color_seq(
                        seq_i,
                        all_natural,
                        aa_dict
                    )
                    print(f"Found {sub}: {seq_i} - {colored_sequence})
                    print(Style.RESET_ALL)

                # summary dataframe for substrates containing kmer
                substrates_with_kmer = pd.DataFrame(index=np.arange(len(subs_q)))
                substrates_with_kmer['Peptide'] = subs_q
                substrates_with_kmer['Sequence'] = seqs_q
            else: # did not find the query kmer
                raise KeyError('K-mer not in dataset. Enter another k-mer')

        else: # no kmers of that length
            raise KeyError(f'No information for kmers of length {kmer_len}. \
                Please use run_kmer_analysis() with length {kmer_len}')

        return substrates_with_kmer

    def find_similar_substrates(self, seq, all_natural, metric, top_k):
        """ Compute similarity between the sequence of interest and substrates
        in the database, using Levenshtein similarity.

            Args:
                seq (str): AA sequence of interest
                all_natural (bool): whether sequence contains only natural
                    AA. if True color-code, if False print uncolored
                metric (str): similarity metric to sort by.
                    2 options: 'Similarity Ratio' or 'Partial Similarity Ratio'
                top_k (int): top_k most similar sequences to print
            Returns:
                sim_m_sorted (pandas df): df of all sequences in the database
                    and their similarity to the sequence of interest
                top_k (int): number of sequences returned
        """
        # Calculate similarity metrics
        sim_m = self.database.copy()
        sim_m = sim_m.iloc[:, 0:3]
        sim_m['Similarity Ratio'] = sim_m.apply(lambda row: \
            paa.substrate.similarity(seq, row['Sequence'])[0], axis=1
        )
        sim_m['Partial Similarity Ratio'] = sim_m.apply(lambda row: \
            paa.substrate.similarity(seq, row['Sequence'])[1], axis=1
        )
        sim_m_sorted = sim_m.sort_values(by=[metric], ascending=False)

        color_seq = paa.substrate.color_seq(seq, all_natural, self.aa_dict)
        print('Queried seq:')
        print(f"{seq} - {color_seq}')
        print(Style.RESET_ALL)

        # Find and print the top_k most similar sequences
        top_k = sim_m_sorted.iloc[:top_k, :]
        for i in np.arange(top_k.shape[0]):
            seq_i = top_k['Sequence'].iloc[i]
            seq_name_i = top_k['Name'].iloc[i]
            seq_aa_i = top_k['Composition'].iloc[i]
            color_seq_i = paa.substrate.color_seq(seq_i, seq_aa_i, self.aa_dict)
            print(f"{seq_name_i}: {seq_i} - {color_seq_i}"")
            print(Style.RESET_ALL)

        return sim_m_sorted, top_k

    def get_similarity_matrix(self, out_dir=False):
        """ Calculate pairwise similarity (Levenshtein) between substrates in
        database.
        Return and plot similarity matrices.

            Args:
                out_dir (str): directory path to save figures
            Returns:
                sim_m (pandas df): df of all subs_list x subs_list.
                    Pairwise Levenshtein distance similarity ratio.
                sim_par_m (pandas df): df of all subs_list x subs_list,
                    Pairwise Partial Levenshtein distance similarity ratio.
                cluster_grid_sim_m (sns clustermap): plot of clustered
                    heatmap of pairwise similarity.
                cluster_grid_sim_par_m (sns clustermap): plot of hierarchically
                    clustered heatmap of pairwise partial similarity.
        """
        subs_list = self.database.iloc[:, 0].to_list()
        seqs_list = self.database['Sequence'].to_list()
        sim_m, sim_par_m, cluster_grid_sim_m, cluster_grid_sim_par_m = \
            paa.substrate.similarity_matrix(subs_list, seqs_list, out_dir)

        return sim_m, sim_par_m, cluster_grid_sim_m, cluster_grid_sim_par_m

    def summarize_kmer(self, kmer_len, top_k, out_dir):
        """ Summarize data of kmers overlapping the database, according to
        their frequency of occurence. Plot histogram of kmer distribution

            Args:
                kmer_len (int): kmer length of interest to query
                top_k (int): the number (k) of top kmers to display
                out_dir (str): directory path for saving figure
            Returns:
                kmer_f_sorted (pandas df): df of kmers sorted by frequency
                    of occurrence in database
                kmer_f_sorted_filtered (pandas df): df of the k top kmers
                    sorted by their frequency in database
            """
        kmer_overlap_q = self.get_kmer_overlap(kmer_len)

        # conduct kmer analysis and plot summary histogram
        kmer_f_sorted, kmer_f_sorted_filtered = paa.substrate.summarize_kmer(
            kmer_overlap_q,
            top_k,
            out_dir
        )

        return kmer_f_sorted, kmer_f_sorted_filtered

    def get_protease_class(self, screen_name):
        """ Get class of proteases in a screen
        Args:
             screen_name (str): screen name to look up protease class of
        Returns:
            protease_class_dict (dict): map of proteases in screen to catalytic
                class
        """
        prot = self.get_screen_proteases(screen_name)
        protease_class_dict = pd.DataFrame(
            data={'Protease': prot}, index=np.arange(len(prot)))
        protease_class_dict['Class'] = protease_class_dict.apply(
            lambda row: paa.protease.classify_protease(row['Protease']), axis=1)

        return protease_class_dict

    def plot_specificity_substrate(self, screen, substrate, out_path,
        threshold=1, close_plot=True, cmap=False):
        """ Plots tissue specificity versus cleavage efficiency.
            Args:
                screen (str) : name of screen in database
                substrate (str): Name of substrate to look up specificity for
                out_path (str): path to store the results
                 threshold (float): lower cut-off for z-scores for labeling
                close_plot (bool): if True, close plots
                cmap (bool): if True, overlay raw intensity values on scatter
        """

        data_matrix = self.raw_screens[screen].transpose()
        subs = self.get_screen_substrates(screen)
        query = substrate
        if query not in subs:
            print('Enter valid substrate in given screen')
            return

        # z-score by column (tissue sample/condition, cleavage efficency)
        cl_z = paa.vis.scale_data(data_matrix)

        # z-score by row (probe, substrate specificity)
        dataT = data_matrix.transpose()
        dataT = paa.vis.scale_data(dataT)
        sp_z = dataT.transpose()
        # display(sp_z)

        # get x and y coordinates for scatterplot
        x = cl_z[query]
        y = sp_z[query]
        prot_col_map = {
            'Metallo': 'g',
            'Serine': 'orange',
            'Aspartic': 'k',
            'Cysteine':'b',
            'Other': 'grey'
        }

        fig, ax = plt.subplots()

        # plot scatter plot with or without colormap
        if cmap:
            raw_prot_vals = data_matrix[query]
            plt.scatter(x, y, c=raw_prot_vals, s=60, edgecolors='grey')
            plt.clim(self.raw_limits[screen][0], self.raw_limits[screen][1])
            cbar = plt.colorbar()
            cbar.set_label('Raw values in ' + screen, fontsize=14)
        else:
            plt.scatter(x, y, s=60)

        plt.xlabel('Cleavage efficiency', fontsize=16)
        plt.ylabel('Specificity', fontsize=16)
        plt.title(query, fontsize=18)
        plt.tight_layout()

        labels = data_matrix.index
        text = []
        for j, txt in enumerate(labels):
            if x[j] > threshold or y[j] > threshold:
                text.append(plt.annotate(
                    txt,
                    (x[j], y[j]),
                    fontsize=12,
                    color=prot_col_map[paa.protease.classify_protease(txt)],
                    weight='bold'
                    )
                )

        adjust_text(text, force_points=4, arrowprops=dict(arrowstyle="-",
            color="k", lw=0.5))
        plt.savefig(os.path.join(out_path, 'specificity_analysis_' +
                                 query + '_' + screen + '.png'))

        if close_plot:
            plt.close()

        return

    def plot_specificity_sample(self, screen, sample, out_path,
        threshold=1, close_plot=True, cmap=False):
        """ Plots sample specificity versus cleavage efficiency.

            Args:
                screen (str) : name of screen in database
                sample (str) : sample name to query (protease, tissue)
                out_path (str): path to store the results
                threshold (float): lower cut-off for z-scores for labeling
                close_plot (bool): if True, close plots
                cmap (bool): if True, overlay raw intensity values on scatter
            Raises:
                ValueError: if sample_name query not found in screen
        """
        data_matrix = self.raw_screens[screen]
        prot = self.get_screen_proteases(screen)
        query = sample_name
        if query not in prot:
            raise ValueError('Enter valid sample name for inputted screen.')

        # z-score by column (tissue sample/condition, cleavage efficency)
        cl_z = paa.vis.scale_data(data_matrix)

        # z-score by row (probe, substrate specificity)
        dataT = data_matrix.transpose()
        dataT = paa.vis.scale_data(dataT)
        sp_z = dataT.transpose()

        # get x and y coordinates for scatterplot
        x = cl_z[query]
        y = sp_z[query]

        plt.figure()

        # plot scatter plot with or without colormap
        if cmap:
            raw_prot_vals = data_matrix[query]
            plt.scatter(x, y, c=raw_prot_vals, s=60, edgecolors='grey')
            plt.clim(self.raw_limits[screen][0], self.raw_limits[screen][1])
            cbar = plt.colorbar()
            cbar.set_label('Raw values in ' + screen, fontsize=14)
        else:
            plt.scatter(x, y, s=60)

        plt.xlabel('Cleavage efficiency', fontsize=16)
        plt.ylabel('Specificity', fontsize=16)
        plt.title(query, fontsize=18)
        plt.tight_layout()

        labels = data_matrix.index
        text = []
        for j, txt in enumerate(labels):
            if x[j] > threshold or y[j] > threshold:
                text.append(plt.annotate(txt, (x[j], y[j]), fontsize=12,
                    weight='bold'))

        adjust_text(text, force_points=4, arrowprops=dict(
            arrowstyle="-", color="k", lw=0.5))
        plt.savefig(os.path.join(out_path, 'specificity_analysis_' +
                                 query + '_' + screen + '.png'))

        if close_plot:
            plt.close()

        return

    def specificity_analysis(self, sample=None, substrate=None, out_path=None,
        threshold=2, close_plot=True, cmap=False):
        """ Perform specificity analysis specificity versus cleavage efficiency
        for all screens in database. Generate resulting plots.
        Consider either a protease or substrate of interest.

            Args:
                screen (str) : name of screen in database
                sample (str) : sample name to query (protease, tissue)
                substrate (str) : substrate name to query
                out_path (str): path to store the results
                threshold (float): lower cut-off for z-scores for labeling
                close_plot (bool): if True, close plots
                cmap (bool): if True, overlay raw intensity values on scatter
            Raises:
                ValueError: if sample or substrate not found in database.
        """
        if (sample is not None) and (substrate is not None):
            print('Please query only 1 protease or 1 substrate at a time')
            return
        elif sample is not None: # sample-wise specificity analysis
            query = sample
            query_df = self.search_protease(query)
            screens_with_query = list(query_df.columns)
            for screen in screens_with_query:
                self.plot_specificity_sample(
                    screen,
                    sample=query,
                    out_path,
                    threshold,
                    close_plot,
                    cmap
                )
        elif substrate is not None: # substrate-wise specificity analysis
            query = substrate
            query_df, query = self.search_substrate(query)
            screens_with_query = list(query_df.columns)
            for screen in screens_with_query:
                self.plot_specificity_substrate(
                    screen,
                    substrate=query,
                    out_path,
                    threshold,
                    close_plot,
                    cmap
                )
        else:
            raise ValueError('Enter a valid sample or substrate')

        return

    def find_proteases(self, prot_list):
        """ Search for list of proteases and return those found in database.

            Args:
                prot_list (list, str) : proteases of interest to look up
            Returns:
                found_proteases (list, str): proteases from list in database
        """
        found_proteases = []
        for protease in prot_list:
            if protease in self.screen_proteases:
                found_proteases.append(protease)
            else:
                print(f'{protease} not found in database')
        return found_proteases

    def find_substrates(self, substrate_list):
        """ Search for list of substrates and return those found in database.

             Args:
                substrate_list (list, str) : names of substrates to look up.
                    Entries of substrate name, not sequence
            Returns:
                found_substrates (list, str): substrates from list in database
        """
        found_substrates = []
        for substrate in substrate_list:
            if substrate in self.screen_substrates:
                found_substrates.append(substrate)
            else:
                sub_name = self.database[self.database['Names'].apply(
                    lambda x: substrate in x)
                ]
                if sub_name.shape[0] == 1:
                    alt_name = sub_name['Name'].item()
                    print(f'Substrate {substrate} is encoded by {alt_name}')
                    found_substrates.append(alt_name)
                else:
                    print(f'Substrate {substrate} not found in database')
        return found_substrates
