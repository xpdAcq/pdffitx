import json
import typing as tp
from pathlib import Path
from tempfile import TemporaryDirectory

from gemmi import cif
from pyobjcryst import loadCrystal
from pyobjcryst.crystal import Crystal

from .tools import get_value


def cif_to_dict(cif_file: str, mmjson: bool = False) -> tp.Generator:
    """Convert cif file to a dictionary."""
    cif_path = Path(cif_file)
    doc = cif.read_file(str(cif_path))
    dct: dict = json.loads(
        doc.as_json(mmjson=mmjson)
    )
    if not mmjson:
        for block_name, block_dct in dct.items():
            block_dct['name'] = block_name
            block_dct['cif_file'] = str(cif_path.absolute())
            yield block_dct
    else:
        yield dct


def to_crystal(dct: dict, keys=("genresults", 0, "stru_str")) -> Crystal:
    """Load the information in the dictionary to a crystasl object. The info is a string of cif file."""
    with TemporaryDirectory() as temp_dir:
        cif_file = Path(temp_dir) / "temp.cif"
        cif_file.write_text(get_value(dct, keys))
        crystal = loadCrystal(str(cif_file))
    return crystal
