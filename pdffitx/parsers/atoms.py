from pathlib import Path
from tempfile import TemporaryDirectory

import ase.io
from ase import Atoms, Atom
from ase.spacegroup import crystal
from diffpy.structure import loadStructure

import pdfstream.parsers.tools as tools


def dict_to_atoms(dct: dict, keys: tuple = ("genresults", 0), **kwargs) -> Atoms:
    """Parse the structure information inside a document into an ase.Atoms object."""
    return stru_dict_to_atoms(
        tools.get_value(dct, keys),
        **kwargs
    )


def stru_dict_to_atoms(dct: dict, **kwargs) -> Atoms:
    """Parse a dictionary of structure information to an ase.Atoms object."""
    return crystal(
        symbols=list(map(atom_dict_to_atom, dct['atoms'])),
        cellpar=lat_dict_to_list(dct["lattice"]),
        spacegroup=dct["space_group"],
        occupancies=get_occ_list(dct["atoms"]),
        **kwargs
    )


def atom_dict_to_atom(dct: dict) -> Atom:
    """Parse a dictionary of atom information to an ase.Atom object."""
    return Atom(
        symbol=tools.only_letter(dct["element"]),
        tag=int(tools.only_digit(dct["name"])),
        position=(dct["x"], dct["y"], dct["z"]),
    )


def lat_dict_to_list(dct: dict) -> list:
    """Make a dictionary of lattice information to a 6-vector [a, b, c, alpha, beta, gamma]."""
    return [
        dct[k] for k in ("a", "b", "c", "alpha", "beta", "gamma")
    ]


def get_occ_list(lst: list) -> list:
    """Get the occupancies list from a list of atom information dictionary."""
    return [
        doc["occ"] for doc in lst
    ]


def dict_to_atoms2(dct: dict, keys: tuple = ("genresults", 0, "stru_str")) -> Atoms:
    """Parser the cif file string in a dictionary to an ase.Atoms.

    Parameters
    ----------
    dct :
        The dictionary that contains the cif file string.

    keys :
        The key chain to find the cif file string in dictionary.

    Returns
    -------
    atoms :
        The atoms object in ase.
    """
    return str_to_atoms(tools.get_value(dct, keys))


def str_to_atoms(stru_str: str) -> Atoms:
    """Convert the cif files string to an ase.Atoms object."""
    with TemporaryDirectory() as temp_dir:
        cif_file = Path(temp_dir) / "temp.cif"
        cif_file.write_text(stru_str)
        stru = loadStructure(str(cif_file))
        stru.write(str(cif_file), format="cif")
        atoms = ase.io.read(str(cif_file))
    return atoms
