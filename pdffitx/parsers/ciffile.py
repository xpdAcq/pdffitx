from pathlib import Path
from tempfile import TemporaryDirectory

from pyobjcryst import loadCrystal
from pyobjcryst.crystal import Crystal

from .tools import get_value


def to_crystal(dct: dict, keys=("genresults", 0, "stru_str")) -> Crystal:
    """Load the information in the dictionary to a crystasl object. The info is a string of cif file."""
    with TemporaryDirectory() as temp_dir:
        cif_file = Path(temp_dir) / "temp.cif"
        cif_file.write_text(get_value(dct, keys))
        crystal = loadCrystal(str(cif_file))
    return crystal
