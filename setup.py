from setuptools import setup

from dwave_networkx import __version__

setup(
    name='dwave_networkx',
    version=__version__,
    packages=['dwave_networkx'],
    install_requires=['networkx>=1.11']
)
