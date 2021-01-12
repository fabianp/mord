from os.path import dirname, join

import numpy as np
from sklearn.datasets.base import Bunch

def load_housing():
    from pandas import read_csv
    """Load and return the Copenhagen housing survey dataset
       (ordinal classification).

    ==============     ==============
    Samples total                1681
    Dimensionality                  3
    Features              categorical
    Targets       ordered categorical
    ==============     ==============

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        and 'DESCR', the full description of the dataset.

    Examples
    --------
    >>> from sklearn.datasets import load_housing
    >>> housing = load_housing()
    >>> print(housing.data.shape)
    (506, 13)
    """
    module_path = dirname(__file__)
    print(module_path)

    fdescr_name = join(module_path, 'descr', 'copenhagen_housing_survey.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data', 'copenhagen_housing_survey.csv')
    data = read_csv(data_file_name)

    '''
    Original data set is formatted as a frequency table,
    but it's more convenient to work with the data
    as having one row per observation, below duplicates
    each obs by index based on the number the frequency ('Freq')
    of appearance
    '''

    # Pandas has deprecated ".ix", so it has to be replaced with "iloc"
    index = np.asarray(range(0, data.shape[0])).\
        repeat(data['Freq'].values)
    data = data.iloc[index,:]
    features = ('Infl', 'Type', 'Cont')

    return Bunch(data=data.loc[:,features],
                 target=data.loc[:,'Sat'],
                 feature_names=features,
                 DESCR=descr_text)
