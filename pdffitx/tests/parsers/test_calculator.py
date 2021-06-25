from diffpy.srreal.bvscalculator import BVSCalculator

from pdffitx.parsers.calculator import bvs_to_xarray


def test_bvs_to_xarray(db):
    stru = db["Ni_stru"]
    calc = BVSCalculator()
    calc(stru)
    arr = bvs_to_xarray(calc)
    print(arr)
