from setuptools import setup, find_packages

from dwave_networkx import __version__, __author__, __description__, __authoremail__

packages = ['dwave_networkx',
            'dwave_networkx.algorithms',
            'dwave_networkx.utils',
            'dwave_networkx.drawing',
            'dwave_networkx.generators']

install_requires = ['networkx>=2.0,<3.0',
                    'decorator>=4.1.0,<5.0.0',
                    'dimod>=0.6.8,<0.8.0']

setup(
    name='dwave_networkx',
    version=__version__,
    author=__author__,
    author_email=__authoremail__,
    description=__description__,
    long_description=open('README.rst').read(),
    url='https://github.com/dwavesystems/dwave_networkx',
    download_url='https://github.com/dwavesystems/dwave_networkx/archive/0.5.0.tar.gz',
    packages=packages,
    license='Apache 2.0',
    install_requires=install_requires
)
