# Copyright 2018 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os

from setuptools import setup

# run from the base path
my_loc = os.path.dirname(os.path.abspath(__file__))
os.chdir(my_loc)

exec(open(os.path.join(".", "dwave", "graphs", "package_info.py")).read())

packages = ['dwave.graphs',
            'dwave.graphs.algorithms',
            'dwave.graphs.drawing',
            'dwave.graphs.generators',
            'dwave.graphs.utils',
            ]

install_requires = ['networkx>=2.4',
                    'dimod>=0.12.5',
                    'numpy>=1.21.3',    # oldest that supports python 3.10
                    ]

python_requires = ">=3.10"

classifiers = [
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3 :: Only',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
    'Programming Language :: Python :: 3.13',
    'Programming Language :: Python :: 3.14',
]

setup(
    name='dwave-graphs',
    version=__version__,
    author=__author__,
    author_email=__authoremail__,
    description=__description__,
    long_description=open('README.rst').read(),
    url='https://github.com/dwavesystems/dwave-graphs',
    download_url='https://github.com/dwavesystems/dwave-graphs/releases',
    packages=packages,
    license='Apache 2.0',
    install_requires=install_requires,
    python_requires=python_requires,
    classifiers=classifiers,
)
