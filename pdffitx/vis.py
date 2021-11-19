import typing
from collections import ChainMap

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


def _get_offset(arr: np.ndarray, delta: float = 0.) -> np.ndarray:
    maxs = np.max(arr, axis=1)
    mins = np.min(arr, axis=1)
    offsets = mins[:-1] - maxs[1:] + delta
    offsets = np.insert(offsets, 0, 0)
    offsets = np.cumsum(offsets)
    return offsets


def get_offset(da: xr.DataArray, delta: float = 0.) -> xr.DataArray:

    return xr.apply_ufunc(
        lambda x: _get_offset(x, delta),
        da,
        input_core_dims=[da.dims],
        output_core_dims=[[da.dims[0]]]
    )


def plot_waterfall(
        ds: xr.Dataset,
        y: str,
        *,
        lab: str = None,
        delta: float = 0.,
        ax: plt.Axes = None,
        line_kws: dict = None,
        hline: bool = True,
        axhline_kws: dict = None,
        lab_xys: typing.List[typing.Tuple[float, float]] = None
) -> plt.Axes:
    """Plot the waterfall lines.

    Parameters
    ----------
    ds :
        The dataset.
    y :
        The name of the y values. It should be a 2d array.
    lab :
        The name of the labels.
    delta :
        Add more or less space between the curves. Unit is the unit of the data.
    ax :
        The Axes to plot. If None, create a new figure.
    line_kws :
        The key words for the line plots.
    hline :
        Whether to plot the zero baselines or not.
    axhline_kws :
        The key words for the axhline plots.
    lab_xys :
        The lilst of xy values for the labels. If None, it will be calculated in best effort.

    Returns
    -------
    The Axes.
    """
    if ax is None:
        _, ax = plt.subplots()
    if line_kws is None:
        line_kws = {}
    if axhline_kws is None:
        axhline_kws = {}
    hue = ds[y].dims[0]
    x = ds[y].dims[1]
    # set the default color and line style
    axhline_kws = dict(
        ChainMap(
            {"color": "gray", "ls": "--"},
            axhline_kws
        )
    )
    # turn off legend if lab is specified
    if lab:
        line_kws["add_legend"] = False
    # get the offsets for the lines
    offset = get_offset(ds[y], delta)
    # shift the lines
    ydata = ds[y] + offset
    ydata.attrs = ds[y].attrs
    # iterate along the hue dimension to add baselines
    if hline:
        n = offset.shape[0]
        for i in range(n):
            ax.axhline(offset[i].item(), **axhline_kws)
    # plot the lines
    ydata.plot.line(**line_kws, hue=hue, ax=ax)
    # add labels
    if lab:
        if lab_xys is None:
            xc = ds[x].max().item() * 0.8
            ycs = (offset.values + ydata.max(dim=x).values) / 2.
            lab_xys = [(xc, yc) for yc in ycs]
        for text, xy in zip(ds[lab].values, lab_xys):
            ax.annotate(str(text), xy, va="center", ha="center")
    # remove the title
    ax.set_title("")
    return ax
