import math
import typing as tp
from tempfile import TemporaryDirectory

from diffpy.srfit.fitbase.fitresults import FitResults, ContributionResults
from diffpy.srfit.pdf import PDFGenerator, DebyePDFGenerator
from diffpy.srfit.structure.srrealparset import SrRealParSet
from diffpy.structure.structure import Structure
from pyobjcryst.crystal import Crystal
from pyobjcryst.spacegroup import SpaceGroup

from pdffitx.modeling import MyRecipe, MyContribution
from pdffitx.modeling.exporter import save_stru

GEN = tp.Union[PDFGenerator, DebyePDFGenerator]


def lattice_to_dict(lattice: SrRealParSet, angunits="rad") -> dict:
    """Convert lattice parameter set to dictionary. If angle is in radian, convert it to degree."""
    dct = dict(zip(lattice.getNames(), lattice.getValues()))
    if angunits == "rad":
        for angle in ('alpha', 'beta', 'gamma'):
            dct[angle] = math.degrees(dct[angle])
    return dct


def atom_to_dict(atom: SrRealParSet) -> dict:
    """Convert atom parameter set to dictionary."""
    dct = dict(zip(atom.getNames(), atom.getValues()))
    dct.update(
        {
            'name': atom.name,
            'element': atom.element
        }
    )
    return dct


def get_space_group_number(phase: SrRealParSet) -> int:
    """Get the space group number of the structure parameter."""
    # if symmetry is not used or the structure object is from diffpy.structure
    if not phase.usingSymmetry() or isinstance(phase.stru, Structure):
        return 1
    stru: Crystal = phase.stru
    # noinspection PyArgumentList
    space_group: SpaceGroup = stru.GetSpaceGroup()
    # noinspection PyArgumentList
    return space_group.GetSpaceGroupNumber()


def phase_to_dict(phase: SrRealParSet) -> dict:
    """Convert structure parameter set to dictionary."""
    return {
        'lattice': lattice_to_dict(phase.getLattice(), angunits=getattr(phase, 'angunits', 'deg')),
        'atoms': [atom_to_dict(atom) for atom in phase.getScatterers()],
        'space_group': get_space_group_number(phase)
    }


def get_genresults(recipe: MyRecipe) -> tp.Generator:
    """Yield the contribution name, generator name and the dictionary expression of the structure."""
    for con_name, con in recipe.contributions.items():
        for gen_name, gen in con.generators.items():
            yield dict(name=gen_name, con_name=con_name, **phase_to_dict(gen.phase))


def conresult_to_dict(result: ContributionResults) -> dict:
    """Convert fit contribution result to dictionary."""
    return {
        'x': result.x.tolist(),
        'y': result.y.tolist(),
        'dy': result.dy.tolist(),
        'ycalc': result.ycalc.tolist(),
        'rw': result.rw,
        'chi2': result.chi2,
        'residual': result.residual
    }


def fitresult_to_dict(result: FitResults) -> dict:
    """Convert fit result to dictionary."""
    return {
        'varnames': result.varnames,
        'varvals': result.varvals.tolist(),
        'varunc': result.varunc,
        'connames': result.connames,
        'convals': result.convals,
        'conunc': result.conunc,
        'fixednames': result.fixednames,
        'fixedvals': result.fixedvals,
        'cov': result.cov.tolist(),
        'residual': result.residual,
        'penalty': result.penalty,
        'chi2': result.chi2,
        'rchi2': result.rchi2,
        'rw': result.rw,
        'precesion': result.precision,
        'derivstep': result.derivstep
    }


def get_conresults(
    conresults: tp.Dict[str, ContributionResults],
    cons: tp.Dict[str, MyContribution]
) -> tp.Generator:
    for name, conresult in conresults.items():
        con = cons[name]
        yield dict(
            name=name,
            eq=con.getEquation(),
            **conresult_to_dict(conresult)
        )


def recipe_to_dict(recipe: MyRecipe) -> dict:
    """Convert the fit result in recipe to a parsers friendly dictionary.

    Parameters
    ----------
    recipe : MyRecipe
        The refined recipe.

    Returns
    -------
    doc : dict
        A nested dictionary containing fitting results, fitted data and the refined structure data.
    """
    result = FitResults(recipe)
    doc = fitresult_to_dict(result)
    doc['conresults'] = list(get_conresults(result.conresults, recipe.contributions))
    doc["genresults"] = list(get_genresults(recipe))
    return doc


def structure_to_str(structure: tp.Union[Structure, Crystal]) -> str:
    """Read the content structure file of the structure and return a string."""
    with TemporaryDirectory() as temp_dir:
        stru_file = save_stru(structure, "temp", temp_dir)
        return stru_file.read_text()


def get_genresults2(recipe: MyRecipe) -> tp.Generator:
    """Yield the dictionary of generator information. The structures are serialized as string."""
    for con_name, con in recipe.contributions.items():
        for gen_name, gen in con.generators.items():
            yield dict(name=gen_name, con_name=con_name, stru_str=structure_to_str(gen.stru))


def recipe_to_dict2(recipe: MyRecipe) -> dict:
    """Convert the fit result in recipe to a parsers friendly dictionary.

    Parameters
    ----------
    recipe : MyRecipe
        The refined recipe.

    Returns
    -------
    doc : dict
        A nested dictionary containing fitting results, fitted data and the refined structure data.
    """
    result = FitResults(recipe)
    doc = fitresult_to_dict(result)
    doc['conresults'] = list(get_conresults(result.conresults, recipe.contributions))
    doc["genresults"] = list(get_genresults2(recipe))
    return doc
