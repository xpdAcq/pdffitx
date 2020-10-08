import numpy as np
from diffpy.srfit.fitbase import ProfileGenerator


class GaussianGenerator(ProfileGenerator):
    """A class for calculating a Gaussian profile.

    Generating a Gaussian is not difficult, as was shown in gaussianrecipe.py.
    Here we create a class that encapsulates this functionality. Placing this
    class in a python module would make it possible to import it and reuse it,
    thereby saving future code writing and debugging.

    The purpose of a ProfileGenerator is to
    1) provide a function that generates a profile signal
    2) organize the Parameters required for the calculation

    Thus, this class overloads the __init__ method to create the necessary
    Parameters for the calculation, and the __call__ method to generate the
    signal.


    """

    def __init__(self, name):
        """Define the generator.

        Note that a ProfileGenerator needs a name passed in the initializer.
        This makes it so the generator can be referenced by name when it is
        part of a FitContribution.

        Here we create the Parameters for the calculation.

        A       --  The amplitude
        x0      --  The center
        sigma   --  The width

        """
        # This initializes various parts of the generator
        ProfileGenerator.__init__(self, name)

        # Here we create new Parameters using the '_newParameter' method of
        # ProfileGenerator. The signature is
        # _newParameter(name, value).
        # See the API for full details.
        self._newParameter('A', 1.0)
        self._newParameter('x0', 0.0)
        self._newParameter('sigma', 1.0)
        return

    def __call__(self, x):
        """Calculate the profile.

        Here we calculate the Gaussian profile given the independent variable,
        x. We will define it as we did in gaussianrecipe.py.

        """
        # First we must get the values of the Parameters. Since we used
        # _newParameter to create them, the Parameters are accessible as
        # attributes by name.
        a = self.A.value
        x0 = self.x0.value
        sigma = self.sigma.value

        # Now we can use them. Note that we imported exp from numpy at the top
        # of the module.
        y = a * np.exp(-0.5 * (x - x0) ** 2 / sigma ** 2)

        # Now return the value.
        return y
