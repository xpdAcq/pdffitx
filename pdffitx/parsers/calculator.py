"""Parse the data in calculator."""
import typing as tp
from collections import defaultdict

from diffpy.srreal.bvscalculator import BVSCalculator
from numpy import ndarray
from xarray import Dataset


def yield_name_attr(obj: object) -> tp.Generator[tp.Tuple[str, tp.Union[float, int, str], int], None, None]:
    """Find float, int, str and ndarray in the attributes. Yield attribute name, value and dimension."""
    for name in dir(obj):
        if not name.startswith("_"):
            attr = getattr(obj, name)
            if isinstance(attr, (float, int, str)):
                yield name, attr, 0
            elif isinstance(attr, ndarray):
                yield name, attr, len(attr.shape)
            else:
                pass


def bvs_to_xarray(calculator: BVSCalculator, dim_name: str = "site") -> Dataset:
    """Convert the BSVCalculator to a xarray dataset."""
    data = dict()
    for name, attr, dim in yield_name_attr(calculator):
        if dim > 0:
            data[name] = (dim_name, attr)
        else:
            data[name] = attr
    coords = defaultdict(lambda: (dim_name, list()))
    for atom in calculator.getStructure():
        for name, attr, dim in yield_name_attr(atom):
            if dim == 0:
                coords[name][1].append(attr)
    return Dataset(data_vars=data, coords=coords)
