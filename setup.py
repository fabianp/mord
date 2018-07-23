from setuptools import setup

setup(
    name='mord',
    version="0.5",
    description='Ordinal regression models',
    long_description=open('README.rst').read(),
    author='Fabian Pedregosa',
    author_email='f@bianp.net',
    url='https://pypi.python.org/pypi/mord',
    packages=['mord'],
    requires=['numpy', 'scipy', "sklearn"],
)
