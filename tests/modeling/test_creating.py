import pytest

import pdfstream.modeling.creating as module


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "name": "C",
            "arange": (0., 10., 0.01),
            "equation": "f0 * G0 + A * sin(r)",
        }
    ],
)
def test_create(data, structures, functions, kwargs):
    recipe = module.create(data=data, structures=structures, functions=functions, **kwargs)
    assert len(recipe.getNames()) == 0
