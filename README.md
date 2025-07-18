# rlfit: Fitting Reinforcement Learning Model to Behavior Data under Bandits

Python package companion to the paper
"Fitting Reinforcement Learning Modelto Behavior Data under Bandits".
This library is collated from the early version code in
[this repository](https://github.com/nrgrp/fit_rl_mab)
which was used for the numerical experiments in the paper.

## Installation

### Using pip

You can install the package via [PyPI](https://pypi.org):

```shell
pip install rlfit
```

### Development setup

We manage dependencies through [uv](https://docs.astral.sh/uv).
Once you have installed uv you can perform the following
commands to set up a development environment:

1. Clone the repository:

    ```shell
    git clone https://github.com/nrgrp/rlfit.git
    cd rlfit
    ```

2. Create a virtual environment and install dependencies:

    ```shell
    make install
    ```

This will:

- Create a Python 3.12 virtual environment.
- Install all dependencies from pyproject.toml.

## Usage

The core module is the `RLFit` class, which was implemented
following the scikit-learn style.
See the [example notebooks](./examples) and the corresponding
paper for some basic usages.
If a development environment is configured, executing

```shell
make jupyter
```

will install and start the jupyter lab.
