import os

def test_installation():
    print("Sucessfully accessing package!")
    return True

def get_data_dir():
    test_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(test_dir, "data")
