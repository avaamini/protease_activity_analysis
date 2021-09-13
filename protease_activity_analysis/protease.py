""" Protease-related functions"""
import pickle
import pandas as pd


def classify_protease(protease):
    """ Map human protease name to protease class

    Args:
        prot (str): name of human protease to be mapped

    Returns:
        class (str): class of protease
    """

    f = open('data/screens/PAA/protease_class_dict.pkl', 'rb')
    class_dict = pickle.load(f)
    # print(class_dict)
    class_mem = None

    for key in class_dict.keys():
        # print(key)
        if protease in class_dict[key]:
            class_mem = key
    if class_mem is None:
        print('Protease ' + protease + ' not found')

    return class_mem


def species_to_species(species_1, species_2, protease):
    """ Map protease from one species to another. Species include Human, Mouse, Chimpanzee and Raet. source: http://degradome.uniovi.es/

    Args:
        species_1 (str): original species
        species_2 (str): new species
        protease (str): protease to be mapped from species_1 to species_2

    Returns:
        mapped_protease (str): mapped protease
    """
    data_path = 'data/screens/PAA/Species_map.xlsx'
    spec_map = pd.read_excel(data_path, header=0)

    keys = spec_map[species_1].to_list()
    vals = spec_map[species_2].to_list()
    spec1_to_seqc2 = dict(zip(keys, vals))

    if protease in spec1_to_seqc2.keys():
        return spec1_to_seqc2[protease]
    else:
        print('Please enter valid protease for ' + species_1)
