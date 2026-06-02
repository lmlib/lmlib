# Installation

## Requirements

- numpy
- scipy
- matplotlib
- numba (optional and needs to be installed manually)

## pip

Using **pip**, install lmlib in the command shell with command

``` console
pip install lmlib
```

To install in **developer mode**, type

``` console
pip install -e lmlib
```

## conda

For **conda** users, the installation runs via the *pip installer* of
conda.

1.  Create a virtual environment (here: venv)

``` console
conda create -n venv
```

2.  Install conda pip

``` console
conda install pip
```

3.  Install lmlib using the path to your virtual environment and install
    the package with

``` console
/anaconda/envs/venv/bin/pip install lmlib
```
