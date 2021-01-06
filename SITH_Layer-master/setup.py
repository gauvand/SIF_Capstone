
#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" comppsychflows setup script """
import sys
from setuptools import setup

# Use setup_requires to let setuptools complain if it's too old for a feature we need
# 30.3.0 allows us to put most metadata in setup.cfg
# 30.4.0 gives us options.packages.find
# 40.8.0 includes license_file, reducing MANIFEST.in requirements
#
# To install, 30.4.0 is enough, but if we're building an sdist, require 40.8.0
# This imposes a stricter rule on the maintainer than the user
# Keep the installation version synchronized with pyproject.toml
SETUP_REQUIRES = ['setuptools >= %s' % ("40.8.0" if "sdist" in sys.argv else "30.4.0")]

# This enables setuptools to install wheel on-the-fly
SETUP_REQUIRES += ['wheel'] if 'bdist_wheel' in sys.argv else []

if __name__ == '__main__':
    # Note that "name" is used by GitHub to determine what repository provides a package
    # in building its dependency graph.
    setup(name='SITH_Layer',
          setup_requires=SETUP_REQUIRES,
          )