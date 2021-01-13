"""A save function to save the fitting results, fitted PDFs and the refined structure in file format from the
FitRecipe."""
import typing as tp
from pathlib import Path

from diffpy.srfit.fitbase import FitContribution, FitResults, FitRecipe
from diffpy.structure import Structure
from pyobjcryst.crystal import Crystal
from pyobjcryst.utils import writexyz

from pdffitx.modeling.core import MyRecipe


def save_fgr(con: FitContribution, base_name: str, folder: str) -> Path:
    """Save fitted PDFs to a four columns txt files with Rw as header.

    Parameters
    ----------
    con : FitContribution
        An arbitrary number of Fitcontribution.

    base_name : str
        The base name for saving. The saving name will be "{base_name}.fgr"

    folder : str
        The folder to save the file.

    Returns
    -------
    fgr_file : Path
        The path to the fgr file.
    """
    fgr_file = Path(folder) / "{}.fgr".format(base_name)
    con.profile.savetxt(fgr_file, header="x ycalc y dy")
    return fgr_file


def write_crystal(stru: Crystal, stru_file: str, fmt: str) -> None:
    """Write out the Crystal object in files."""
    if fmt == "cif":
        with Path(stru_file).open("w") as f:
            stru.CIFOutput(f)
    elif fmt == "xyz":
        writexyz(stru, stru_file)
    else:
        raise ValueError("Unknown format: {}. Allow: 'cif', 'xyz'.".format(fmt))
    return


def save_stru(
    stru: tp.Union[Crystal, Structure], base_name: str, folder: str, fmt: str = "cif"
) -> Path:
    """Save refined structure.

    Parameters
    ----------
    stru : Crystal or Structure
        A structure object.

    base_name : str
        The base name for saving. The saving name will be "{base_name}.{fmt}."

    folder : str
        The folder to save the file.

    fmt : str
        The format of the structure file. Options are "cif" and "xyz". Default "cif".

    Returns
    -------
    stru_file : Path
        The path to the saved files.
    """
    stru_file = Path(folder) / "{}.{}".format(base_name, fmt)
    if isinstance(stru, Crystal):
        write_crystal(stru, str(stru_file), fmt)
    else:
        stru.write(str(stru_file), format=fmt)
    return stru_file


def save_res(recipe: FitRecipe, base_name: str, folder: str) -> Path:
    """Save the fitting results.

    Parameters
    ----------
    recipe : FitRecipe
        The refined recipe.

    base_name : str
        The base name of the result file. The file name will be "{base_name}.res".

    folder : str
        The folder to save the fitting result file.

    Returns
    -------
    res_file : Path
        The path to the fitting result file.
    """
    res_file = Path(folder) / "{}.res".format(base_name)
    res = FitResults(recipe)
    res.saveResults(str(res_file))
    return res_file


def save(
    recipe: MyRecipe, base_name: str, folder: str, stru_fmt: str = "cif"
) -> tp.Tuple[Path, tp.List[Path], tp.List[Path]]:
    """Save the results of the refined recipe.

    The fitting results will be saved in ".res" file. It is a text file of the fitting parameter values and
    variance. The fitted PDF data will be saved in ".fgr" files. It is a four columns text file. The columns are
    (x, ycalc, y, dy). The refined structure will be saved in text files. The format is determined by "stru_fmt".

    Parameters
    ----------
    recipe : MyRecipe
        The refined recipe.

    base_name : str
        The base name for the files. The files will be names as "{base_name}.res", "{base_name}_{
        contribution.name}.fgr" and "{base_name}_{contribution.name}_{generator.name}.stru".

    folder : str
        The folder to save the result files.

    stru_fmt : str
        The structure format. See :func:`~.exporter.save_stru`.

    Returns
    -------
    res_file : Path
        The path to the fitting result file.

    fgr_files : List[Path]
        A list of the paths to the fitted PDF files.

    stru_files : List[Path]
        A list of the paths to the structure files.
    """
    res_file = save_res(recipe, base_name, folder)
    fgr_files = []
    stru_files = []
    for con_name, con in recipe.contributions.items():
        fgr_file = save_fgr(
            con,
            "{}_{}".format(base_name, con_name),
            folder
        )
        fgr_files.append(fgr_file)
        for gen_name, gen in con.generators.items():
            stru_file = save_stru(
                gen.stru,
                "{}_{}_{}".format(base_name, con_name, gen_name),
                folder,
                fmt=stru_fmt
            )
            stru_files.append(stru_file)
    return res_file, fgr_files, stru_files
