from setuptools import setup, find_packages

setup(
    name='mord',
    version="0.6",
    description='Ordinal regression models',
    long_description=open('README.rst').read(),
    author='Fabian Pedregosa',
    author_email='f@bianp.net',
    url='https://pypi.python.org/pypi/mord',
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    include_package_data=True,
    requires=['numpy', 'scipy', "sklearn"],
)
