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

exec(open(os.path.join(".", "dwave", "plugins", "networkx", "package_info.py")).read())

packages = ['dwave_networkx',
            'dwave.plugins.networkx',
            'dwave.plugins.networkx.algorithms',
            'dwave.plugins.networkx.utils',
            'dwave.plugins.networkx.drawing',
            'dwave.plugins.networkx.generators',
            ]

install_requires = ['networkx>=2.0,<3.0',
                    'decorator>=4.1.0,<5.0.0',
                    'dimod>=0.8.0',
                    ]

python_requires = ">=3.5"

classifiers = [
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3 :: Only',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
]

setup(
    name='dwave-networkx',
    version=__version__,
    author=__author__,
    author_email=__authoremail__,
    description=__description__,
    long_description=open('README.rst').read(),
    url='https://github.com/dwavesystems/dwave-networkx',
    download_url='https://github.com/dwavesystems/dwave-networkx/releases',
    packages=packages,
    license='Apache 2.0',
    install_requires=install_requires,
    python_requires=python_requires,
    classifiers=classifiers,
    zip_safe=False,
)
