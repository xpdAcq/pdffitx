import typing
import functools

from dataclasses import dataclass
import xarray as xr
import numpy as np
from scipy.signal import find_peaks
from diffpy.srreal.bondcalculator import BondCalculator
from pyobjcryst.crystal import Crystal
from pyobjcryst import loadCrystal


class PeakMactherError(Exception):
    pass


@dataclass
class PeakMactherConfig:
    """The configuration for the PeakMatcher.
    """
    rwidth: typing.Sequence[float]
    rwlen: float
    rdistance: float
    rel_height: float
    height: typing.Optional[typing.Sequence[float]] = None
    threshold: typing.Optional[typing.Sequence[float]] = None
    rstep: float = 0.01


class PeakMatcher:
    """Match the peaks with the atomic distance.
    """

    def __init__(self, config: PeakMactherConfig):
        self._config = config
        self._bonds = BondCalculator()

    def fit(self, data: xr.Dataset, crystal: Crystal) -> xr.Dataset:
        """Fit the data. Return the results of the matched peaks.

        Parameters
        ----------
        data
        crystal

        Returns
        -------

        """
        dists = self._find_dists(data, crystal)
        peaks = self._find_peaks(data)
        self._macth_dists(peaks, dists)
        return xr.merge([dists, peaks])

    @staticmethod
    def _check_data(data: xr.Dataset) -> None:
        """Check the format of the data.

        Parameters
        ----------
        data

        Returns
        -------

        """
        if "r" not in data:
            raise PeakMactherError("No variable named `r` in data.")
        if "G" not in data:
            raise PeakMactherError("No variable named `G` in data.")
        n_dim = data["G"].ndim
        if n_dim != 1:
            raise PeakMactherError("The number of the dimension of `G` ndim != 1. ndim = {}".format(n_dim))
        return

    def _find_dists(self, data: xr.Dataset, crystal: Crystal) -> xr.Dataset:
        """Set the rmin and rmax.

        Parameters
        ----------
        data
        crystal

        Returns
        -------

        """
        r = data["r"].values
        rmin, rmax = float(r[0]), float(r[-1])
        self._bonds.rmin = rmin
        self._bonds.rmax = rmax
        self._bonds(crystal)
        attrs0 = {"units": r"$mathrm{\AA}$", "source": "crystal"}
        attrs1 = {"source": "crystal"}
        dim = ["dist_id"]
        return xr.Dataset(
            {
                "atom0": (dim, self._bonds.types0, attrs1),
                "atom1": (dim, self._bonds.types1, attrs1),
                "rmin": ([], self._bonds.rmin, attrs0),
                "rmax": ([], self._bonds.rmax, attrs0),
                "distance": (dim, self._bonds.distances, attrs0)
            }
        )

    def _find_peaks(self, data: xr.Dataset) -> xr.Dataset:
        """Fit find the peaks in the data.

        Parameters
        ----------
        data

        Returns
        -------

        """
        rstep = self._config.rstep
        width = [round(w / rstep) for w in self._config.rwidth]
        distance = round(self._config.rdistance / rstep)
        wlen = round(self._config.rwlen / rstep)
        y = data["G"].values
        idxs, props = find_peaks(
            y,
            width=width,
            distance=distance,
            wlen=wlen,
            height=self._config.height,
            threshold=self._config.threshold,
            rel_height=self._config.rel_height
        )
        x = data["r"].values
        left_ips = np.asarray(props["left_ips"], dtype=int)
        right_ips = np.asarray(props["right_ips"], dtype=int)
        attrs = {"units": r"$mathrm{\AA}$", "source": "data"}
        dim = ["peak_id"]
        return xr.Dataset(
            {
                "middle": (dim, x[idxs], attrs),
                "left": (dim, x[left_ips], attrs),
                "right": (dim, x[right_ips], attrs)
            }
        )

    @staticmethod
    def _macth_dists(peaks: xr.Dataset, dists: xr.Dataset) -> None:
        """Match the bond distances in the range of the peaks. Modify the first dataset inplace.

        Parameters
        ----------
        peaks
        dists

        Returns
        -------

        """
        dists = dists["distance"].values  # get the data and rename
        start = 0
        n = peaks.sizes["peak_id"]
        ls, rs = [], []
        for i in range(n):
            sel_peaks = peaks.isel({"peak_id": i})
            left = sel_peaks["left"].item()
            right = sel_peaks["right"].item()
            l = np.searchsorted(dists[start:], left, "left")
            r = np.searchsorted(dists[l:], right, "right")
            start = r  # move the start idx to reduce search range for the next search
            ls.append(l)
            rs.append(r)
        peaks["left_dist_id"] = (["peak_id"], ls)
        peaks["right_dist_id"] = (["peak_id"], ls)
        attrs = {"source": "matching"}
        peaks["left_dist_id"].attrs = attrs
        peaks["right_dist_id"].attrs = attrs
        return


def get_atomic_pairs(result: xr.Dataset, pattern: str = r"{}-{}") -> typing.List[typing.List[str]]:
    """Get the atomic pairs.

    Parameters
    ----------
    result
    pattern

    Returns
    -------

    """
    pairses = []
    ls = result["left_dist_id"].values
    rs = result["right_dist_id"].values
    atoms0 = result["atom0"].values
    atoms1 = result["atom1"].values
    for l, r in zip(ls, rs):
        pairs = []
        for a0, a1 in zip(atoms0[l:r], atoms1[l:r]):
            pairs.append(pattern.format(a0, a1))
        pairses.append(pairs)
    return pairses


def get_distances(result: xr.Dataset) -> typing.List[typing.List[float]]:
    """Get the atomic distances.

    Parameters
    ----------
    result

    Returns
    -------

    """
    reses = []
    ls = result["left_dist_id"].values
    rs = result["right_dist_id"].values
    dists = result["distance"].values
    for l, r in zip(ls, rs):
        res = []
        for d in dists[l:r]:
            res.append(d)
        reses.append(res)
    return reses
