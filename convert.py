import numpy as np
import pickle
import gzip
import json
from functools import singledispatch


@singledispatch
def to_serializable(val):
    """Used by default."""
    return str(val)


# @to_serializable.register(np.float32)
# def ts_float32(val):
#     """Used if *val* is an instance of numpy.float32."""
#     return np.float64(val)


@to_serializable.register(np.ndarray)
def ts_ndarray(val):
    """Used if *val* is an instance of numpy.ndarray."""
    return val.tolist()

# json.loads(json.dumps({'pi': np.float32(3.1415)}, default=to_serializable))
# json.loads(json.dumps({'pi': np.array([3.1415])}, default=to_serializable))


def convert_to_numpy(input_list):
    if isinstance(input_list, list):
        return np.array([convert_to_numpy(item) for item in input_list])
    else:
        return input_list

# nested_list = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
# numpy_array = convert_to_numpy(nested_list)


def dump_pkl_to_json():
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        data_pkl = pickle.load(f, encoding="latin1")
    with open('mnist.json', 'w') as json_out:
        json.dump(data_pkl, json_out, default=to_serializable)


def load_json_to_list():
    with open('mnist.json', 'r') as json_in:
        data_list = json.load(json_in)
    return data_list


def convert_lists_to_numpy(data_list):
    data_0 = [convert_to_numpy(data_list[0][0]), convert_to_numpy(data_list[0][1])]
    data_1 = [convert_to_numpy(data_list[1][0]), convert_to_numpy(data_list[1][1])]
    data_2 = [convert_to_numpy(data_list[2][0]), convert_to_numpy(data_list[2][1])]
    return (tuple(data_0), tuple(data_1), tuple(data_2))


def convert_pkl_to_numpy():
    print("Dumping json from pkl")
    dump_pkl_to_json()
    print("Loading json to list")
    data_list = load_json_to_list()
    print("Converting Lists to Numpy")
    return convert_lists_to_numpy(data_list)
