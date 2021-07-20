#!/usr/bin/env python
"""A Nose plugin to support IPython doctests.
"""
#upload to pip
#pip install .
#python setup.py sdist
#twine upload dist/Forecast_Checker-0.1.tar.gz

from setuptools import setup

setup(name='Forecast_Checker',
      version='0.1',
      url='https://github.com/dstarkey23/Forecast_Checker',
      author='drds.gh1@gmail.com',
      author_email='drds.gh1@gmail.com',
      description = 'Evaluate Forecast model performance vs step size',
      license = 'Apache',
      packages=['Forecast_Checker',],
      entry_points = {},
      install_requires=['pandas','numpy','matplotlib','statsmodels']
      )

