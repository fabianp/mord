from setuptools import setup
import mord

setup(
    name='mord',
    version=mord.__version__,
    description='Ordinal regression models',
    long_description=open('README.rst').read(),
    author='Fabian Pedregosa',
    author_email='f@bianp.net',
    url='https://pypi.python.org/pypi/mord',
    packages=['mord'],
    requires=['numpy', 'scipy'],
)
