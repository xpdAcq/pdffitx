"""Parse fit results."""
import typing as tp

import pandas as pd
import xarray as xr

from .tools import get_value


def to_dataframe(
        dct: dict,
        column: tuple = ("conresults", 0, "name"),
        index: tuple = ("varnames",),
        data: tuple = ("varvals",),
        index_name: str = None
):
    """Convert the fitting results to a dataframe"""
    arr = to_xarray(dct, column, index, data)
    df = arr.to_dataframe()
    df.rename_axis(index_name, inplace=True)
    return df


def to_xarray(
        dct: dict, name: tuple = ("conresults", 0, "name"),
        coords: tuple = ("varnames",), data: tuple = ("varvals",), dim: str = "parameter"
) -> xr.DataArray:
    """Convert the fitting results to xarray data array.

    Parameters
    ----------
    dct :
        A dictionary of fitting results.

    name :
        The keys to find the name for the data array.

    coords :
        The keys to find the coordinates values.

    data :
        The keys to find the data values.

    dim :
        The name of the dimension of the coordinates.

    Returns
    -------
    array :
        A one-dimensional data array.
    """
    return xr.DataArray(
        data=get_value(dct, data),
        coords={dim: get_value(dct, coords)},
        dims=[dim],
        name=get_value(dct, name)
    )


def rename_rule(name: str) -> str:
    """A conventional rule to rename the parameter name to latex version."""
    words = name.split("_")
    if "scale" in words:
        return "scale"
    if "delta2" in words:
        return r"$\delta_2$ ($\mathrm{\AA}^2$)"
    if "delta1" in words:
        return r"$\delta_1$ ($\mathrm{\AA}$)"
    for word in ("a", "b", "c"):
        if word in words:
            return word + r" ($\mathrm{\AA}$)"
    for word in ("alpha", "beta", "gamma"):
        if word in words:
            return rf"$\{word}$" + " (deg)"
    for word in (
            'Uiso', 'U11', 'U12', 'U13', 'U21', 'U22', 'U23', 'U31', 'U32', 'U33',
            'Biso', 'B11', 'B12', 'B13', 'B21', 'B22', 'B23', 'B31', 'B32', 'B33',
    ):
        if word in words:
            return rf"{word[0]}$_{{{word[1:]}}}$({words[1]})" + r" ($\mathrm{\AA}^2$)"
    for word in ("x", "y", "z"):
        if word in words:
            return rf"{word}({words[1]})" + r" ($\mathrm{\AA}$)"
    for word in ("psize", "psig", "sthick", "thickness", "radius"):
        if word in words:
            return rf"{word}" + r" ($\mathrm{\AA}$)"
    return " ".join(words[1:])


def to_latex(*ndfs: tp.Tuple[str, pd.DataFrame]) -> str:
    """Conver the dataframe to latex tubular with multicolumn row.

    Parameters
    ----------
    ndfs :
        The arbitrary number of (name of the multicolumn row, the data frame)

    Returns
    -------
    latex :
        A latex string.
    """
    if len(ndfs) == 0:
        return ""
    if len(ndfs) == 1:
        _, df = ndfs[0]
        return df.to_latex(escape=True)
    total = list()
    head, df = ndfs[0]
    total.extend(to_lines(df, head)[:-3])
    for head, df in ndfs[1:-1]:
        total.extend(to_lines(df, head)[3:-3])
    head, df = ndfs[-1]
    total.extend(to_lines(df, head)[3:])
    return "\n".join(total)


def to_lines(df: pd.DataFrame, head: str, escape=False) -> tp.List[str]:
    """
    Convert the results data frame to a list of lines in the string of latex table.

    Parameters
    ----------
    df :
        The data frame of the results. Index are the parameters and columns are the samples.

    head :
        The content of multicolumn row.

    escape
        When set to False prevents from escaping latex special characters in column names. Default False.

    Returns
    -------
    lines
        A list of lines in the latex table.
    """
    origin_str = df.to_latex(escape=escape)
    lines = origin_str.split('\n')
    row = r"\multicolumn{" + str(df.shape[1]) + r"}{l}{" + head + r"}"
    lines = lines[:4] + [row, r"\midrule"] + lines[4:]
    return lines
