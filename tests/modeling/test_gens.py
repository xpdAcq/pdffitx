import numpy as np

from pdfstream.modeling.gens import GaussianGenerator


def test_GaussianGenerator():
    gen = GaussianGenerator("Gaussian")
    for a in ('A', 'x0', 'sigma'):
        assert hasattr(gen, a)
    assert np.array_equal(np.ones(3), gen(np.zeros(3)))
