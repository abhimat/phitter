% Phitter Installation

# Installation

## Dependencies

Phitter relies heavily on [PHOEBE](https://www.phoebe-project.org) and [SPISEA](https://spisea.readthedocs.io/en/latest/). Both packages are required for using Phitter.

We recommend first installing PHOEBE 2.4+ in a new Python environment. Full installation instructions for PHOEBE are [here](https://www.phoebe-project.org/install).

SPISEA can then be installed. Installation instructions for SPISEA are [here](https://spisea.readthedocs.io/en/latest/getting_started.html).

## Install Phitter via Github
Clone the [Phitter repository from Github](https://github.com/abhimat/phitter). Next, add the path to the cloned Phitter repository into the `$PYTHONPATH` variable into your shell's corresponding `.zshrc`, `.bash_profile`, or `.bashrc` file:
```sh
export PYTHONPATH=$PYTHONPATH:/local/path/to/phitter/
```

## Install Phitter via pip & Conda
Support for installation via `pip` and `conda` is not currently implemented, but it is planned for the future 😅!

## Test Phitter installation
Installation of phitter can be tested via the following line in Python:

```py
from phitter.calc import model_obs_calc
```

```sh
/Users/abhimat/Software/miniforge3/envs/phoebe_py38/lib/python3.8/site-packages/pysynphot/locations.py:345: UserWarning: Extinction files not found in /Users/abhimat/models/cdbs/extinction
  warnings.warn('Extinction files not found in %s' % (extdir, ))
```

A user warning output about missing extinction files may be generated from pysynphot via SPISEA. The functionality being warned about is not used by Phitter or SPISEA, and can be safely ignored.
