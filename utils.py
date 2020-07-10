""" Utilities for scripts """
import numpy as np
import pandas as pd
import itertools
import os

## Global variables: reporter names for 14 and 20 plex MS panels
PLEX_14 = ["1UR3_01", "2UR3_02", "3UR3_03", "4UR3_04", "5UR3_05", "6UR3_06",
       "7UR3_07", "8UR3_08", "9UR3_10", "1UR3_11", "2UR3_13", "3UR3_15",
       "4UR3_16", "5UR3_18"]
RENAMED_14 = ["PP01", "PP02", "PP03", "PP04", "PP05", "PP06", "PP07", "PP08",
          "PP09", "PP10", "PP11", "PP12", "PP13", "PP14"]

PLEX_20 = ["1UR3_01", "2UR3_02", "3UR3_03", "4UR3_04", "5UR3_05", "6UR3_06",
            "7UR3_07","8UR3_08", "9UR3_10", "1UR3_11", "2UR3_13", "3UR3_15",
            "4UR3_16", "5UR3_18", "6UR3_20", "1UR3_09", "2UR3_12", "3UR3_14",
            "4UR3_17", "5UR3_19"]
RENAMED_20 = ["PP01", "PP02", "PP03", "PP04", "PP05", "PP06", "PP07", "PP08",
              "PP09", "PP10", "PP11", "PP12", "PP13", "PP14", "PP15", "PP16",
              "PP17", "PP18", "PP19", "PP20"]

def get_data_dir():
    test_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(test_dir, "data")

def get_output_dir():
    test_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(test_dir, "outputs")
