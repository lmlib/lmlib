from setuptools import setup, find_packages

setup(
    name='lmlib',
    version='2.0.2',
    description='A Model-Based Signal Processing Library Working With Windowed Linear State-Space and Polynomial Signal Models.',
    author='Reto Wildhaber, Frédéric Waldmann, Luca Fleischmann, Christof Baeriswyl',
    author_email='reto.wildhaber@fhnw.ch, frederic.waldmann@fhnw.ch',
    url='http://lmlib.ch',
    download_url='https://pypi.python.org/pypi/lmlib',
    license='MIT',
    packages=find_packages(include=['lmlib', 'lmlib.*']),
    install_requires=[
            'numpy',
            'matplotlib',
            'scipy'],
    extras_require={'jit-backend': ['numba']},
    include_package_data = True,
    python_requires='>=3'
)
