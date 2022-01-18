import xarray as xr
from pdffitx.peakmatcher import PeakMactherConfig, PeakMatcher, get_distances, get_atomic_pairs


def test_PeakMatcher(db):
    r, g = db["Ni_gr"]
    r, g = r[100:500], g[100:500]
    crystal = db["Ni_stru"]
    data = xr.Dataset({"G": (["r"], g)}, coords={"r": r})
    config = PeakMactherConfig(rwidth=[0.2, 1.0], rwlen=1.5, rdistance=0.4, rel_height=0.5)
    pm = PeakMatcher(config)
    result = pm.fit(data, crystal)
    dists = get_distances(result)
    assert len(dists) > 0
    pairs = get_atomic_pairs(result)
    assert len(pairs) > 0
