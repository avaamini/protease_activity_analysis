""" Protease - substrate database."""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import os

class SubstrateDatabase(object):

    def __init__(self, file_list):
        self.database = {}
        self.file_list = file_list

        self.substrates = {}
        self.proteases = {}
        self.sequences = {}

        self.unique_substrates = set()
        self.unique_proteases = set()
        self.unique_sequences = set()

        for f in file_list:
            # TODO: incorporate unified name mapping, as well as sequences
            data, name, substrates, proteases = self.load_dataset(f, z_score=True)
            # TODO: have a way to load the sequences
            self.database[name] = data

            self.substrates[name] = substrates
            self.proteases[name] = proteases
            # self.sequences[name] = sequences

            self.unique_substrates.update(substrates)
            self.unique_proteases.update(proteases)
            # self.unique_sequences.update(sequences)

    def load_dataset(self, file_path, z_score):
        """ Load dataset from a csv file

        Args:
            file_path (str): path to the screening data file
            z_score (bool): whether or not to z-score data for standardization
        """
        # TODO: incorporate the unified name mapping

        data = pd.read_csv(file_path)
        file_name = os.path.basename(file_path).split('.csv')[0]

        data = data.groupby('Substrate').mean().reset_index()
        substrates = list(data['Substrate'])
        proteases = list(data.iloc[:,1:].columns)

        #Correct for possible spaces
        proteases = [item.replace(' ', '') for item in proteases]
        data.columns = ['Substrate'] + proteases  #Change names in df

        data = data.set_index('Substrate')

        if z_score:
            data = (data - data.mean()) / data.std(ddof=0)

        return data, file_name, substrates, proteases

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

        for screen in screens:
            if protease in self.proteases[screen]:
                print(f"{protease} found in {screen}")

                # get cleavage data for the protease found in particular screen
                protease_cleavage_dict[screen] = self.database[screen][protease]

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

        for screen in screens:
            if substrate in self.substrates[screen]:
                print(f"{substrate} found in {screen}")

                # get cleavage data for the protease found in particular screen
                substrate_cleavage_dict[screen] = self.database[screen].loc[substrate]

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
                os.path.join(out_dir, f"{query}_{z_threshold}_{top_k}_individual.csv")
            )

            top_k_overall.to_csv(
                os.path.join(out_dir, f"{query}_{z_threshold}_{top_k}_overall.csv")
            )

        return top_k_individual, top_k_overall
