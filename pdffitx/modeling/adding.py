"""Add variables to recipe."""
import typing as tp

from diffpy.srfit.pdf import PDFGenerator, DebyePDFGenerator

from .fitobjs import MyRecipe, MyContribution

G = tp.Union[PDFGenerator, DebyePDFGenerator]


def initialize(
    recipe: MyRecipe,
    scale: bool = True,
    delta: tp.Union[str, None] = "2",
    lat: tp.Union[str, None] = "s",
    adp: tp.Union[str, None] = "a",
    xyz: tp.Union[str, None] = "s",
    params: tp.Union[None, str, tp.List[str]] = "a"
) -> MyRecipe:
    """Initialize a single-contribution recipe with the variables.

    The variables will be constrained, created and added according to the mode indicated by the arguments. If an
    argument is None, nothing will be done. It is assumed that the recipe only has one contribution and all
    variables in the generators in that contribution will be initialized in the same way. To constrain specific
    contribution and generator, please use `~pdfstream.modeling.adding.add_con_vars` and
    `~pdfstream.modeling.adding.add_gen_vars`.

    The name for each parameter follows the rule of "{generator or function name}_{variable name}". For the
    B-factors and coordinates variables, the variable name follows the rule
    "{atom name or element name}_{Biso, B11, B22, B33, x, y, z}".

    Each parameter in generator is tagged by three tags: the catagory of the paramter ("scale", "delta",
    "lat", "adp", "xyz"), the name of the generator (e. g. "G0") and their union (e. g. "G0_lat"). Each parameter
    in the contribution is tagged by the name of its function if the parameter is named by
    "{function name}_{...}".

    Parameters
    ----------
    recipe :
        A recipe with a single contribution and generators.

    scale :
        Whether to add the scale of the generator. Default True.

    delta :
        If "1", add delta1 parameter.
        If "2", add delta2 parameter.
        If None, do nothing.

    lat :
        If "s", constrain the lattice parameters by space group and add the independent variables.
        If "a", add all the lattice paramters.
        If None, do nothing.
        Default "s".

    adp :
        If "a", add all the Biso parameter.
        If the structure is `~pyobjcryst.crystal.Crystal`, this step means add all the Biso of unique atoms.
        If "e", constrain the Biso by elements, add the independent Biso.
        If "s", constrain the B-tensor by space group, add all the independent diagonal terms like B11, B22, B33
        or Biso.
        If None, do nothing.
        Default "a".

    xyz :
        If "s", constrain the coordinates of atoms by space group and add independent coordinates.
        If "a", add all coordinates of atoms.
        If None, do nothing.
        Default "s".

    params :
        If "a", add all the parameters in the equation and characteristic functions like "psize".
        If list of str, add all the parameters whose names are in the list.
        If None, do nothing.
        Default "a".

    Returns
    -------
    recipe :
        The initialized recipe. The operation is done inplace so the recipe is the same as the input.
    """
    add_con_vars(recipe, params=params)
    add_gen_vars(recipe, scale=scale, delta=delta, lat=lat, adp=adp, xyz=xyz)
    return recipe


def add_con_vars(
    recipe: MyRecipe,
    name: tp.Union[str, None] = None,
    params: tp.Union[None, str, tp.List[str]] = "a"
):
    """Add variables at contribution level.

    The name is the name of the contribution. If None, all the contribution will be searched and added.
    """
    if not name:
        for con_name in recipe.contributions.keys():
            add_con_vars(recipe, con_name, params)
        return
    con = recipe.contributions[name]
    add_params(recipe, con, params)
    return


def add_params(recipe: MyRecipe, con: MyContribution, params: tp.Union[None, str, tp.List[str]]) -> None:
    """Add contribution-level parameters in the contribution."""
    if not params:
        return
    args = {
        arg.name: arg
        for eq in con.eqfactory.equations
        if eq.name == "eq"
        for arg in eq.args
        if arg.name != con.xname
    }
    if params == "a":
        pars = args.values()
    else:
        pars = [args[p] for p in params]
    for par in pars:
        recipe.addVar(par, tag=par.name.split("_")[0])
    return


def add_gen_vars(
    recipe: MyRecipe,
    names: tp.Tuple[str, str] = None,
    scale: bool = True,
    delta: tp.Union[str, None] = "2",
    lat: tp.Union[str, None] = "s",
    adp: tp.Union[str, None] = "a",
    xyz: tp.Union[str, None] = "s",
) -> None:
    """Add parameters at generator level.

    The names are (contribution name, generator name). If None, all generators will be constrained and added.
    """
    if not names:
        for con_name, con in recipe.contributions.items():
            for gen_name in con.generators.keys():
                add_gen_vars(
                    recipe, names=(con_name, gen_name), scale=scale, delta=delta, lat=lat, adp=adp, xyz=xyz
                )
        return
    gen = recipe.contributions[names[0]].generators[names[1]]
    add_scale(recipe, gen, scale)
    add_delta(recipe, gen, delta)
    add_lat(recipe, gen, lat)
    add_adp(recipe, gen, adp)
    add_xyz(recipe, gen, xyz)
    return


def add_scale(recipe: MyRecipe, gen: G, scale: bool = True) -> None:
    """Add the scale of the generator."""
    if not scale:
        return
    recipe.addVar(
        gen.scale,
        value=0.,
        name="{}_scale".format(gen.name),
        tags=["{}_scale".format(gen.name), "scale", gen.name]
    ).boundRange(
        lb=0.
    )
    return


def add_delta(recipe: MyRecipe, gen: G, delta: tp.Union[str, None]) -> None:
    """Add the delta parameter of the generator."""
    if not delta:
        return
    if delta == "1":
        par = gen.delta1
    elif delta == "2":
        par = gen.delta2
    else:
        raise ValueError("Unknown delta: {}. Allowed: delta1, delta2.".format(delta))
    recipe.addVar(
        par,
        value=0.,
        name="{}_{}".format(gen.name, par.name),
        tags=["{}_delta".format(gen.name), "delta", gen.name]
    ).boundRange(
        lb=0.
    )
    return


def add_lat(recipe: MyRecipe, gen: G, lat: tp.Union[str, None]) -> None:
    """Add the lattice parameters of the phase."""
    if not lat:
        return
    if lat == "s":
        pars = gen.phase.sgpars.latpars
    elif lat == "a":
        pars = gen.phase.getLattice()
    else:
        raise ValueError("Unknown lat: {}. Allowed: sg, all.".format(lat))
    for par in pars:
        recipe.addVar(
            par,
            name="{}_{}".format(gen.name, par.name),
            tags=["lat", gen.name, "{}_lat".format(gen.name)]
        ).boundRange(
            lb=0.
        )
    return


def add_adp(
    recipe: MyRecipe, gen: G, adp: tp.Union[str, None],
    symbols: tp.Tuple[str] = ("Biso", "B11", "B22", "B33")
) -> None:
    """Add the atomic displacement parameter of the phase."""
    if not adp:
        return
    atoms = gen.phase.getScatterers()
    if adp == "e":
        elements = set((atom.element for atom in atoms))
        dct = dict()
        for element in elements:
            dct[element] = recipe.newVar(
                "{}_{}_Biso".format(gen.name, bleach(element)),
                value=0.05,
                tags=["adp", gen.name, "{}_adp".format(gen.name)]
            )
        for atom in atoms:
            recipe.constrain(atom.Biso, dct[atom.element])
        return
    if adp == "a":
        pars = [atom.Biso for atom in atoms]
        names = ["{}_Biso".format(bleach(atom.name)) for atom in atoms]
    elif adp == "s":
        pars = gen.phase.sgpars.adppars
        names = [
            rename_by_atom(par.name, atoms) for par in pars
            if par.name.split("_")[0] in symbols
        ]
    else:
        raise ValueError("Unknown adp: {}. Allowed: element, sg, all.".format(adp))
    for par, name in zip(pars, names):
        recipe.addVar(
            par,
            name="{}_{}".format(gen.name, name),
            value=par.value if par.value != 0. else 0.05,
            tags=["adp", gen.name, "{}_adp".format(gen.name)]
        ).boundRange(
            lb=0.
        )
    return


def add_xyz(recipe: MyRecipe, gen: G, xyz: tp.Union[str, None]) -> None:
    """Add the coordinates of the atoms in the phase."""
    if not xyz:
        return
    atoms = gen.phase.getScatterers()
    if xyz == "s":
        pars = gen.phase.sgpars.xyzpars
        names = [rename_by_atom(par.name, atoms) for par in pars]
    elif xyz == "a":
        pars, names = list(), list()
        for atom in atoms:
            pars.append(atom.x)
            names.append("{}_x".format(atom.name))
            pars.append(atom.y)
            names.append("{}_y".format(atom.name))
            pars.append(atom.z)
            names.append("{}_z".format(atom.name))
    else:
        raise ValueError("Unknown xyz: {}. Allowed: s, a.".format(xyz))
    for par, name in zip(pars, names):
        recipe.addVar(
            par,
            name="{}_{}".format(gen.name, name),
            tags=["xyz", gen.name, "{}_xyz".format(gen.name)]
        )
    return


def rename_by_atom(name: str, atoms: list) -> str:
    """Rename of the name of a parameter by replacing the index of the atom in the name by the label of
    the atom and revert the order of coordinates and atom name.

    Used for the space group constrained parameters. For example, "x_0" where atom index 0 is Ni will become
    "Ni0_x" after renamed. If the name can not renamed, return the original name.
    """
    parts = name.split("_")
    if len(parts) > 1 and parts[1].isdigit() and -1 < int(parts[1]) < len(atoms):
        parts[1] = atoms[int(parts[1])].name
        parts = parts[::-1]
    return "_".join(parts)


def bleach(s: str):
    """Bleach the string of the atom names and element names.

    Replace '+' with 'p', '-' with 'n'. Strip all the characters except the number and letters.
    """
    return ''.replace("+", "p").replace("-", "n").join((c for c in s if c.isalnum()))
