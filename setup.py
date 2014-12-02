from distutils.core import setup
import numpy as np
import setuptools

setup(name='mord',
    version='0.1',
    description='Ordinal Regression algorithms',
    author='Fabian Pedregosa',
    author_email='f@bianp.net',
    url='',
    packages=['mord'],
    requires = ['numpy', 'scipy'],
)
