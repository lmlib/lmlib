.. _install:

Install
-------

pip
^^^

Using **pip**, install lmlib in the command shell with command
::
   pip install lmlib

To install in **developer mode**, type
::
   pip install -e lmlib


conda
^^^^^

For **conda** users, the installation runs via the *pip installer* of conda.

1. Create a virtual environment (here: venv)
   ::
      conda create -n venv

2. Install conda pip
   ::
      conda install pip

3. Install lmlib using the path to your virtual environment and install the package with
   ::
      /anaconda/envs/venv/bin/pip install lmlib
