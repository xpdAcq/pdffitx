import pytest

from pdffitx.parsers.fitrecipe import recipe_to_dict2
from pdffitx.parsers.fitresult import to_dataframe, to_latex, rename_rule


def test_to_latex(recipe_two_strus):
    expect = r"""\begin{tabular}{lr}
\toprule
{} &          test \\
\midrule
\multicolumn{1}{l}{Ni0}
\midrule
G_scale &  1.510116e-08 \\
\midrule
\multicolumn{1}{l}{Ni1}
\midrule
G_scale &  1.510116e-08 \\
\bottomrule
\end{tabular}
"""
    dct = recipe_to_dict2(recipe_two_strus)
    df = to_dataframe(dct, ("conresults", 0, "name"))
    latex = to_latex(("Ni0", df), ("Ni1", df))
    assert latex == expect


@pytest.mark.parametrize(
    "name, expect",
    [
        ("G0_scale", "scale"),
        ("G0_delta1", r"$\delta_1$ ($\mathrm{\AA}$)"),
        ("G0_delta2", r"$\delta_2$ ($\mathrm{\AA}^2$)"),
        ("G0_a", r"a ($\mathrm{\AA}$)"),
        ("G0_alpha", r"$\alpha$ (deg)"),
        ("G0_Ni0_Biso", r"B$_{iso}$(Ni0) ($\mathrm{\AA}^2$)"),
        ("G0_Ni0_x", r"x(Ni0) ($\mathrm{\AA}$)"),
        ("f0_psize", r"psize ($\mathrm{\AA}$)")
    ]
)
def test_rename_rule(name, expect):
    real = rename_rule(name)
    assert real == expect
