import multiprocessing
from typing import Tuple, Union, Dict

import matplotlib.pyplot as plt
import numpy as np
from diffpy.srfit.fitbase import Profile, FitContribution
from diffpy.srfit.fitbase.parameter import ParameterProxy
from diffpy.srfit.pdf import PDFGenerator, DebyePDFGenerator
from diffpy.srfit.structure.diffpyparset import DiffpyStructureParSet
from diffpy.srfit.structure.objcrystparset import ObjCrystCrystalParSet
from diffpy.srfit.structure.sgconstraints import constrainAsSpaceGroup
from matplotlib.axes import Axes
from scipy.optimize import least_squares

from pdfstream.modeling.fitobjs import MyParser, ConConfig, GenConfig, MyRecipe, MyContribution
from pdfstream.visualization.main import visualize

__all__ = [
    'make_profile',
    'make_generator',
    'make_contribution',
    'make_recipe',
    'fit',
    'plot',
    'constrainAsSpaceGroup',
    'sgconstrain',
    'cfconstrain',
    'sgconstrain_all',
    'cfconstrain_all',
    'get_sgpars'
]


# functions used in fitting
def make_profile(parser: MyParser, fit_range: Tuple[float, float, float]) -> Profile:
    """
    Make a Profile, parse data file to it and set its calculation range.

    Parameters
    ----------
    parser
        The parser with parsed data from the data source.

    fit_range
        The tuple of (rmax, rmin, dr) in Angstrom.

    Returns
    -------
    profile
        The Profile with the parsed data and the calculation range.
    """
    profile = Profile()
    profile.loadParsedData(parser)
    rmin, rmax, rstep = fit_range
    profile.setCalculationRange(rmin, rmax, rstep)
    return profile


def make_generator(genconfig: GenConfig) -> Union[PDFGenerator, DebyePDFGenerator]:
    """
    Build a generator according to the information in the GenConfig.

    Parameters
    ----------
    genconfig : GenConfig
        A configuration instance for generator building.

    Returns
    -------
    generator: PDFGenerator or DebyePDFGenerator
        A generator built from GenConfig.
    """
    generator = DebyePDFGenerator(genconfig.name) if genconfig.debye else PDFGenerator(genconfig.name)
    generator.setStructure(genconfig.structure, periodic=genconfig.structure)
    ncpu = genconfig.ncpu
    if ncpu:
        pool = multiprocessing.Pool(ncpu)
        generator.parallel(ncpu, mapfunc=pool.imap_unordered)
    return generator


def make_contribution(conconfig: ConConfig, xname: str = "r") -> MyContribution:
    """
    Make a FitContribution according to the ConConfig.

    Parameters
    ----------
    conconfig : ConConfig
        The configuration instance for the FitContribution.

    xname : str
        The name of the independent variable. Default 'r'.

    Returns
    -------
    contribution : MyContribution
        The FitContribution built from ConConfig.
    """
    contribution = MyContribution(conconfig.name)

    fit_range = conconfig.fit_range
    profile = make_profile(conconfig.parser, fit_range)
    contribution.setProfile(profile, xname=xname)

    for genconfig in conconfig.genconfigs:
        generator = make_generator(genconfig)
        contribution.addProfileGenerator(generator)

    for base_line in conconfig.baselines:
        contribution.addProfileGenerator(base_line)

    for function in conconfig.funconfigs:
        name = function.name
        func_type = function.func
        argnames = function.argnames
        contribution.registerFunction(func_type, name, argnames)

    contribution.setEquation(conconfig.eq)
    contribution.setResidualEquation(conconfig.res_eq)

    return contribution


def make_recipe(*conconfigs: ConConfig) -> MyRecipe:
    """
    Make a FitRecipe based on single or multiple ConConfig.

    Parameters
    ----------
    conconfigs
        The configurations of single or multiple FitContribution.

    Returns
    -------
    recipe
        MyRecipe built from ConConfigs.
    """
    recipe = MyRecipe()

    for conconfig in conconfigs:
        contribution = make_contribution(conconfig)
        recipe.addContribution(contribution, conconfig.weight)

    recipe.clearFitHooks()

    return recipe


def fit(recipe: MyRecipe, **kwargs) -> None:
    """
    Fit the data according to recipe. parameters associated with fitting can be set in kwargs.

    Parameters
    ----------
    recipe
        MyRecipe to fit.

    kwargs
        Parameters in fitting. They are
            verbose: how much information to print. Default 1.
            values: initial value for fitting. Default get from recipe.
            bounds: two list of lower and upper bounds. Default get from recipe.
            xtol, gtol, ftol: tolerance in least squares. Default 1.E-5, 1.E-5, 1.E-5.
            max_nfev: maximum number of evaluation of residual function. Default None.
    """
    values = kwargs.get("values", recipe.values)
    bounds = kwargs.get("bounds", recipe.getBounds2())
    verbose = kwargs.get("verbose", 1)
    xtol = kwargs.get("xtol", 1.E-5)
    gtol = kwargs.get("gtol", 1.E-5)
    ftol = kwargs.get("ftol", 1.E-5)
    max_nfev = kwargs.get("max_fev", None)
    least_squares(recipe.residual, values, bounds=bounds, verbose=verbose, xtol=xtol, gtol=gtol, ftol=ftol,
                  max_nfev=max_nfev)
    return


def plot(contribution: FitContribution) -> Axes:
    """
    Plot the fits for all FitContributions in the recipe.

    Parameters
    ----------
    contribution : FitContribution
        The FitRecipe.

    Returns
    -------
    ax : Axes
        The axes that has the plot.
    """
    r = contribution.profile.x
    g = contribution.profile.y
    gcalc = contribution.profile.ycalc
    gdiff = g - gcalc
    data = np.stack([r, g, gcalc, gdiff])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = visualize(
        data,
        ax=ax,
        mode='fit',
        legends=["observed", "calculated", "zero", "residual"],
        label="gr"
    )
    ax.set_title(contribution.name)
    plt.show(block=False)
    return ax


def cfconstrain(recipe: MyRecipe, con_name: str, param_names: list = None, dv: Dict[str, float] = None,
                bounds: Dict[str, tuple] = None) -> Dict[str, ParameterProxy]:
    """Add parameters in the contribution.

    Add parameters in the Characteristic functions in the FitContribution into the MyRecipe.
    Return the added variables in a dictionary.

    Parameters
    ----------
    recipe : MyRecipe
        The recipe to add the parameters

    con_name : str
        The name of the FitContribution where the parameters are.

    param_names : list
        The name of the parameter to be added in the recipe. If None, add all parameters.

    dv : dict
        The path to the .csv file contains the fitting results or the dictionary of values.
        If None, use par.value for any parameters as default value.

    bounds : dict
        The mapping from the name of the variable to the tuple of bounds (min, max). Defulat (0, +inf).

    Returns
    -------
    variables : dict
        The dictionary mapping from the name of the variable to the variable itself.
    """
    if dv is None:
        dv = dict()
    if bounds is None:
        bounds = dict()
    variables = dict()
    con: MyContribution = getattr(recipe, con_name)
    if param_names is None:
        # get all the parameter in the contribution except the independent variable
        pars = {
            arg
            for eq in con.eqfactory.equations
            if eq.name == "eq"
            for arg in eq.args
            if arg.name != con.xname
        }
    else:
        pars = {getattr(con, param_name) for param_name in param_names}
    for par in pars:
        variables[par.name] = recipe.addVar(par, value=dv.get(par.name, par.value), tag="cf").boundRange(
            *bounds.get(par.name, (-np.inf, np.inf)))
    return variables


def cfconstrain_all(recipe: MyRecipe, dv: dict = None, bounds: dict = None):
    """Constrain all the parameters in registered functions and string equations."""
    variables = dict()
    for con_name in recipe.contributions:
        variables.update(
            cfconstrain(recipe, con_name, dv=dv, bounds=bounds)
        )
    return variables


def sgconstrain(recipe: MyRecipe, con_name: str, gen_name: str, sg: Union[int, str] = None,
                dv: Union[str, Dict[str, float]] = None, bounds: Dict[str, tuple] = None,
                add_xyz: bool = True) -> Dict[str, ParameterProxy]:
    """Constrain the generator by space group.

    The constrained parameters are scale, delta2, lattice parameters, ADPs and xyz coordinates. The lattice
    constants and xyz coordinates are constrained by space group while the ADPs are constrained by elements.
    All paramters will be added as '{par.name}_{gen.name}'. The parameters tags are scale_{gen.name},
    delta2_{gen.name}, lat_{gen.name}, adp_{gen.name}, xyz_{gen.name}. Return the added variables in a
    dictionary.

    Parameters
    ----------
    recipe
        The recipe to add variables.

    con_name
        The name of the FitContribution where the PDFGenerator is in. If None, get the first contribution.
        Default None.

    gen_name
        The name of the PDFGenerator to constrain. If None, constrain the first generator in contribution.

    sg
        The space group. The expression can be number or name. If the structure is Crystal object, use internal
        constrain.

    dv
        The path to the .csv file contains the fitting results or the dictionary of values.
        If None, the following values will be used:
        type, initiate value, range, tag
        scale, 0, (0, inf), scale_{gen.name}
        delta2, 0, (0, inf), delta2_{gen.name}
        lat, par.value, (0, 2 * par.value), lat_{gen.name}
        adp, 0.05, (0, inf), adp_{gen.name}
        xyz, par.value, None, xyz_{gen.name}

    bounds
        The mapping from the name of the variable to the tuple of the arguments for the bounding function.

    add_xyz
        Whether to constrain xyz coordinates. If True, xyz will be added as fixed variable. Default True.

    Returns
    -------
    variables
        The dictionary mapping from the name of the variable to the variable itself.
    """
    # initiate variables
    variables = dict()
    # the default of variables
    if dv is None:
        dv = dict()
    # the bounds
    if bounds is None:
        bounds = dict()
    # get FitContribution and PDFGenerator
    con: MyContribution = getattr(recipe, con_name)
    gen: Union[PDFGenerator, DebyePDFGenerator] = getattr(con, gen_name)
    # add scale
    name = f'scale_{gen.name}'
    variables[name] = recipe.addVar(gen.scale, name=name, value=dv.get(name, 0.)).boundRange(
        *bounds.get(name, (0., np.inf)))
    # add delta2
    name = f'delta2_{gen.name}'
    variables[name] = recipe.addVar(gen.delta2, name=name, value=dv.get(name, 0.)).boundRange(
        *bounds.get(name, (0., np.inf)))
    # constrain by spacegroup
    sgpars = get_sgpars(gen.phase, sg)
    # add latpars
    for par in sgpars.latpars:
        name = f'{par.name}_{gen.name}'
        variables[name] = recipe.addVar(par, name=name, value=dv.get(name, par.value),
                                        tag=f'lat_{gen.name}').boundRange(*bounds.get(name, (0., 2. * par.value)))
    # constrain adps
    atoms = gen.phase.getScatterers()
    elements = set([atom.element for atom in atoms])
    adp = dict()
    for element in elements:
        name = f'Biso_{only_alpha(element)}_{gen.name}'
        variables[name] = adp[element] = recipe.newVar(name, value=dv.get(name, 0.05),
                                                       tag=f'adp_{gen.name}').boundRange(
            *bounds.get(name, (0, np.inf)))
    for atom in atoms:
        recipe.constrain(getattr(atom, 'Biso'), adp[atom.element])
    # add xyzpars
    if add_xyz:
        for par in sgpars.xyzpars:
            name = f'{par.name}_{gen.name}'
            variables[name] = recipe.addVar(par, name=name, value=dv.get(name, par.value),
                                            tag=f'xyz_{gen.name}', fixed=True).boundRange(
                *bounds.get(name, (-np.inf, np.inf)))
    return variables


def get_sgpars(parset: Union[ObjCrystCrystalParSet, DiffpyStructureParSet], sg: Union[int, str] = None):
    """Constrain the structure by space group and get the independent parameters."""
    if isinstance(parset, ObjCrystCrystalParSet):
        if sg is not None:
            print(
                "ObjCrystCrystalParSet does not accept explicit space group constrain. "
                "Implicit space group is used."
            )
        sgpars = parset.sgpars
    elif isinstance(parset, DiffpyStructureParSet):
        if sg is None:
            sg = 'P1'
            print(
                "No explicit space group for DiffpyStructureParSet. "
                "Use 'P1' symmetry."
            )
        sgpars = constrainAsSpaceGroup(
            parset, sg, constrainadps=False
        )
    else:
        raise ValueError(
            "{} does not allow space group constrain.".format(type(parset))
        )
    return sgpars


def sgconstrain_all(recipe: MyRecipe, dv: dict = None, bounds: dict = None) -> dict:
    """Use space group to constrain all the generators in the recipe. See sgconstrain for details.

    Parameters
    ----------
    recipe : MyRecipe
        The recipe where variables will be added.

    dv : dict
        The keys are the names of variables and the values are the initial value for optimization.

    bounds : dict
        The keys are the names of variables and the keys are the tuple of start and end or single value for window.
    """
    variables = dict()
    for con_name, con in recipe.contributions.items():
        for gen_name, gen in con.generators.items():
            if isinstance(gen, (DebyePDFGenerator, PDFGenerator)):
                variables.update(
                    sgconstrain(
                        recipe, con_name, gen_name, dv=dv, bounds=bounds
                    )
                )
    return variables


def only_alpha(s: str):
    """Remove all characters other than alphabets. Use to get a valid variable name."""
    return ''.join((c for c in s if c.isalpha()))
