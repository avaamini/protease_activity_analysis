""" Protease - substrate database."""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import os

class SubstrateDatabase(object):

    def __init__(self, data_files, sequence_file, names_file=None):
        self.database = {}
        self.file_list = data_files

        self.substrates = {}
        self.proteases = {}

        self.unique_substrates = set()
        self.unique_proteases = set()

        for f in data_files:

            data, name, substrates, proteases = self.load_dataset(f)
            self.database[name] = data

            self.substrates[name] = substrates
            self.proteases[name] = proteases

            self.unique_substrates.update(substrates)
            self.unique_proteases.update(proteases)

        # load sequence information
        sequence_info = self.load_sequence_info(sequence_file)
        self.sequences = sequence_info
        self.unique_sequences = list(set(sequence_info['Sequence']))

        # Mapping of sequence names to alternative names/descriptors
        name_mapping = self.load_substrate_names(names_file)
        self.name_map = name_mapping

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
        data.columns = ['Name'] + proteases  #Change names in df

        data = data.set_index('Name')

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
        seq_data = seq_data.set_index('Name')
        return seq_data

    def get_screen_substrates(self, screen_name):
        """ Get all the substrates from a particular screen.

        Args:
            screen_name (str): the screen of interest.

        Returns:
            screen_substrates (list, str): substrate token names from screen of
                interest
        """
        return self.substrates[screen_name]

    def get_screen_proteases(self, screen_name):
        """ Get all the proteases from a particular screen.

        Args:
            screen_name (str): the screen of interest.

        Returns:
            screen_proteases (list, str): protease token names from screen of
                interest
        """
        return self.proteases[screen_name]

    def get_screen_data(self, screen_name):
        """ Get data from a particular screen.

        Args:
            screen_name (str): the screen of interest.

        Returns:
            screen_df (df): data from screen of interest
        """
        return self.database[screen_name]

    def search_protease(self, protease, out_dir=None, z_threshold=None):
        """ Return df of substrates and their cleavage by a given protease

        Args:
            protease (str): protease of interest
            out_dir (str): if specified, directory for writing data
            z_threshold (float): upper bound z-score cutoff for substrate return
        Returns:
            protease_cleavage_df (pandas df): zscores for protease against all
                substrates found in the database
        """
        protease_cleavage_dict = {}
        screens = self.proteases.keys()

        protease_found = False
        for screen in screens:
            if protease in self.proteases[screen]:
                print(f"{protease} found in {screen}")

                # get cleavage data for the protease found in particular screen
                protease_cleavage_dict[screen] = self.database[screen][protease]
                protease_found = True

        if not protease_found:
            raise ValueError("Protease not found in database. Please try again.")

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

    def search_substrate(self, substrate, out_dir=None, z_threshold=None):
        """ Return df of proteases and their cleavage of a given substrate

        Args:
            substrate (str): substrate of interest
            out_dir (str): if specified, directory for writing data
            z_threshold (float): upper bound z-score cutoff for substrate return
        Returns:
            protease_cleavage_df (pandas df): zscores for substrate against all
                proteases found in the database
        """
        substrate_cleavage_dict = {}
        screens = self.substrates.keys()

        substrate_found = False
        for screen in screens:
            if substrate in self.substrates[screen]:
                print(f"{substrate} found in {screen}")

                # get cleavage data for the protease found in particular screen
                substrate_cleavage_dict[screen] = self.database[screen].loc[substrate]
                substrate_found = True

        if not substrate_found:
            raise ValueError("Substrate not found in database. Please try again.")

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
            top_k_screen.loc[:,'Source'] = screen
            individual_dfs.append(top_k_screen)

            top_k_screen_names = list(top_k_screen.index)
            print(f"Top hits for {query} in {screen}:")
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
