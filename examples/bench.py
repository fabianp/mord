from sklearn import linear_model, cross_validation, datasets, grid_search
import mord
import numpy as np

for n_samples in range(40, 200, 20):
    X, y = datasets.make_classification(n_samples=n_samples, n_features=10000,
        n_classes=5, n_informative=5)
    clf1 = linear_model.LogisticRegressionCV(solver='lbfgs',
                                            multi_class='multinomial')

    clf2 = grid_search.GridSearchCV(mord.LogisticIT(), {'alpha': np.logspace(
        -3, 3, 10)})
    cv = cross_validation.StratifiedShuffleSplit(y, test_size=0.3)
    print(cross_validation.cross_val_score(clf1, X, y, cv=cv).mean())
    print(cross_validation.cross_val_score(clf2, X, y, cv=cv).mean())
    print()
