import typing as tp

import numpy as np

import pdffitx.parsers.tools as tools


def dicts_to_array(dcts: tp.Iterable[dict], keys: tuple = ("conresults", 0),
                   data_keys: tuple = ("x", "y", "ycalc"), **kwargs) -> np.array:
    """Convert a series of dictionaries of str and list pairs to a numpy array.

    Parameters
    ----------
    dcts : Iterable[dict]
        The a series of dictionaries, each containing a sub-dictionary of str and list pairs.

    keys : tuple
        A key chain to find the sub-dictionary of str and list pairs.

    data_keys : tuple
        A dictionary of data keys. Their corresponding value will be added into the numpy array.

    kwargs : dict
        The kwargs for the `:func:~numpy.stack`.

    Returns
    -------
    array : ndarray
        The numpy array of data. Axis 0 is corresponding to the dictionaries and Axis 1 is correspongding to a
        list in the dictionary.
    """
    return np.stack(
        [
            dict_to_array(dct, keys=keys, data_keys=data_keys, **kwargs) for dct in dcts
        ],
        **kwargs
    )


def dict_to_array(dct: dict, keys: tuple = tuple(), data_keys: tuple = ("x", "y", "ycalc"), **kwargs) -> np.array:
    """Convert a dictionary of str and list pairs to a numpy array.

    Parameters
    ----------
    dct : dict
        The dictionary that contains a sub-dictionary of str and list pairs.

    keys : tuple
        A key chain to find the sub-dictionary of str and list pairs.

    data_keys : tuple
        A dictionary of data keys. Their corresponding value will be added into the numpy array.

    kwargs : dict
        The kwargs for the `:func:~numpy.stack`.

    Returns
    -------
    array : ndarray
        The numpy array of data. Each row corresponding to a list in the dictionary.
    """
    data_dct = tools.get_value(dct, keys)
    return np.stack(
        [
            data_dct[key] for key in data_keys
        ],
        **kwargs
    )
