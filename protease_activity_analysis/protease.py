""" Protease-related helper functions."""
import pickle
import pandas as pd

def classify_protease(protease):
    """ Map human protease name to protease class

    Args:
        protease (str): name of human protease to be mapped

    Returns:
        class_mem (str): class membership of query protease
    """

    f = open('data/screens/PAA/protease_class_dict.pkl', 'rb')
    class_dict = pickle.load(f)
    # print(class_dict)
    class_mem = None

    for key in class_dict.keys():
        if protease in class_dict[key]:
            class_mem = key
    if class_mem is None:
        class_mem = 'Other'
        print('Protease ' + protease + ' not found')

    return class_mem


def species_to_species(species_1, species_2, protease):
    """ Map protease name from one species to another.
    Species include Human, Mouse, Chimpanzee, Rat.
    Source: http://degradome.uniovi.es/

    Args:
        species_1 (str): original species
        species_2 (str): new species
        protease (str): protease to be mapped from species_1 to species_2

    Returns:
        mapped_protease (str): protease homolog in the new species (species_2)
    Raises:
        KeyError: if query protease is not valid for species_1
    """
    data_path = 'data/screens/PAA/Species_map.xlsx'
    spec_map = pd.read_excel(data_path, header=0)

    keys = spec_map[species_1].to_list()
    vals = spec_map[species_2].to_list()
    species1_to_species2 = dict(zip(keys, vals))

    if protease in species1_to_species2.keys():
        mapped_protease = species1_to_species2[protease]
        return mapped_protease
    else:
        raise KeyError('Please enter valid protease for ' + species_1)
