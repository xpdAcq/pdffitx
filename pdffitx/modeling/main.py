import inspect
import typing as tp

from diffpy.srfit.fitbase import FitResults
from diffpy.srfit.pdf import PDFGenerator
from diffpy.structure import Structure
from matplotlib.axes import Axes
from pyobjcryst.crystal import Crystal

from pdfstream.modeling.fitfuncs import (
    make_recipe, sgconstrain_all, cfconstrain_all, fit, plot, sgconstrain, cfconstrain
)
from pdfstream.modeling.fitobjs import MyRecipe, GenConfig, ConConfig, MyParser, FunConfig, MyContribution
from pdfstream.modeling.saving import save

__all__ = [
    'multi_phase',
    'optimize',
    'GenConfig',
    'ConConfig',
    'MyParser',
    'MyRecipe',
    'report',
    'view_fits',
    'fit_calib',
    'FIT_RANGE',
    'Crystal',
    'Structure',
    'save'
]

FIT_RANGE = tp.Tuple[float, float, float]
STRU = tp.Union[Crystal, Structure]
PHASE = tp.Union[STRU, tp.Tuple[tp.Callable, STRU]]


def fit_calib(
    stru: Crystal, data: MyParser, fit_range: FIT_RANGE, ncpu: int = None
) -> MyRecipe:
    """The fit the pdf of the calibration. Get the qdamp and qbraod.

    Parameters
    ----------
    stru : Crystal
        The structure of calibration material.

    data : MyParser
        The parser that contains the pdf data.

    fit_range : tuple
        The rmin, rmax and rstep in the unit of angstrom.

    ncpu : int
        The number of cpu used in parallel computing. If None, no parallel computing.

    Returns
    -------
    recipe : MyRecipe
        The refined recipe.
    """
    recipe = multi_phase([stru], data, fit_range, ncpu=ncpu)
    con: MyContribution = next(iter(recipe.contributions.values()))
    gen: PDFGenerator = next(iter(con.generators.values()))
    recipe.addVar(gen.qdamp, tag='qparams')
    recipe.addVar(gen.qbroad, tag='qparams')
    optimize(
        recipe,
        tags=['scale_G0', 'lat_G0', 'adp_G0', 'delta2_G0', 'qparams'],
        verbose=0
    )
    print("\n")
    report(recipe)
    print("\n")
    view_fits(recipe)
    return recipe


def multi_phase(
    phases: tp.Iterable[PHASE],
    data: MyParser,
    fit_range: tp.Tuple[float, float, float],
    values: dict = None,
    bounds: dict = None,
    ncpu: int = None,
    cf_params: tp.List[str] = None,
    sg_params: tp.Dict[str, tp.Union[int, str]] = None,
    add_eq: str = None
) -> MyRecipe:
    """Make the recipe of a multiphase crystal pdf refinement.

    The function assumes that the structure is the :func:`~pyobjcryst.crystal.Crystal`. The parameters
    are taged as 'scale', 'delta2', 'lat', 'adp', 'xyz' (optional) with the suffix '_{the name of the
    generator}'. The unit depends on the structure loaded in the generator.

    Parameters
    ----------
    phases : Iterable.
        An iterable of structures or tuple of characteristic function and structure.

    data : MyParser
        A parser with parsed data.

    fit_range : tuple
        The rmin, rmax and rstep in the unit of angstrom.

    ncpu : int
        The number of cpu used in parallel computing. If None, no parallel computing.

    values : dict
        The the dictionary of default values.
        If None, the following values will be used:
        tag     initiate value      range
        scale   0                   (0, inf)
        delta2  0                   (0, inf)
        lat     par.value           (0, 2 * par.value)
        adp     0.05                (0, inf)
        xyz     par.value           None

    bounds : dict
        The mapping from the name of the variable to the bounds.

    cf_params : list
        A list of parameter names in characteristic functions to be added in the recipe. If cf_params = None,
        all the parameters in all the contributions will be added. If cf_params = [], no parameters will
        be added into recipe.

    sg_params : dict
        The keys are the generator names and the values are the space group notations or numbers.
        If a value is None, the generator will be constrained according to the implicit space group information
        in its phase. If sg_params = None, all the generators will be constrained and the parameters will be
        added to the recipe. If sg_params = {}, no generators will be contrained and no parameters will be added.

    add_eq : str
        Additional equation in the fitting.

    Returns
    -------
    recipe : MyRecipe
        The recipe with symmetrically constrained variables and without refinement.
    """
    genconfigs, funconfigs, eqs = list(), list(), dict()
    for i, phase in enumerate(phases):
        if isinstance(phase, tuple):
            # attenuated structure
            cf, stru = phase
        else:
            # pure crystal structure
            cf, stru = None, phase
        gname = "G{}".format(i)
        genconfigs.append(
            GenConfig(name=gname, structure=stru, ncpu=ncpu)
        )
        eq = gname
        if cf is not None:
            fname = "f{}".format(i)
            funconfigs.append(
                FunConfig(name=fname, func=cf, argnames=add_suffix(cf, fname))
            )
            eq += " * " + fname
        eqs.update({gname: eq})
    total_eq = " + ".join(eqs.values())
    if add_eq:
        total_eq = " + ".join((total_eq, add_eq))
    conconfig = ConConfig(name='multi_phase', eq=total_eq, parser=data, fit_range=fit_range,
                          genconfigs=genconfigs, funconfigs=funconfigs)
    recipe = make_recipe(conconfig)
    if cf_params is None:
        cfconstrain_all(
            recipe, dv=values, bounds=bounds
        )
    else:
        cfconstrain(recipe, conconfig.name, param_names=cf_params)
    if sg_params is None:
        sgconstrain_all(
            recipe, dv=values, bounds=bounds
        )
    else:
        for gen_name, space_group in sg_params.items():
            sgconstrain(recipe, conconfig.name, gen_name, sg=space_group)
    return recipe


def optimize(recipe: MyRecipe, tags: tp.List[tp.Union[str, tp.Iterable[str]]], **kwargs) -> MyRecipe:
    """First fix all variables and then free the variables one by one and fit the recipe.

    Parameters
    ----------
    recipe
        The recipe to fit.

    tags
        The tags of variables to free. It can be single tag or a tuple of tags.

    kwargs
        The kwargs of the 'fit'.
    """
    verbose = kwargs.pop('verbose', 0)
    if verbose > 0:
        print(f"Start {recipe.name} with all parameters fixed.")
    recipe.fix('all')
    for n, tag in enumerate(tags):
        if isinstance(tag, str):
            if verbose > 0:
                print("Free {} ...".format(tag))
            recipe.free(tag)
        else:
            if verbose > 0:
                print(
                    "Free {} ...".format(
                        ", ".join(tag)
                    )
                )
            recipe.free(*tag)
        fit(recipe, verbose=verbose, **kwargs)
    return recipe


def report(recipe: MyRecipe) -> FitResults:
    """Print out the fitting result.

    Parameters
    ----------
    recipe : MyRecipe
        The recipe after refinement.

    Returns
    -------
    res : FitResults
        The object contains the fit results.
    """
    res = FitResults(recipe)
    res.printResults()
    return res


def view_fits(recipe: MyRecipe) -> tp.List[Axes]:
    """View the fit curves. Each FitContribution will be a plot.

    Parameters
    ----------
    recipe : MyRecipe
        The recipe after refinement.

    Returns
    -------
    axes : a list of Axes
        The plots of the fits.
    """
    axes = []
    for con in recipe.contributions.values():
        ax = plot(con)
        axes.append(
            ax
        )
    return axes


def add_suffix(func: tp.Callable, suffix: str) -> tp.List[str]:
    """Add the suffix to the argument names starting at the second the argument. Return the names"""
    args = inspect.getfullargspec(func).args
    return list(
        map(
            lambda arg: '{}_{}'.format(arg, suffix) if arg != "r" else arg,
            args
        )
    )
