""" Utilities for scripts """
import numpy as np
import pandas as pd
import itertools
import os

def get_data_dir():
    test_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(test_dir, "data")

def get_output_dir():
    test_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(test_dir, "outputs")
