import os
import sys

import dwave_networkx as nx

version = nx.__version__

tag = os.getenv('CIRCLE_TAG')

if tag != version:
    sys.exit("Git tag: {}, expected: {}".format(tag, version))
