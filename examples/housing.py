from sklearn import linear_model, cross_validation, datasets,\
    metrics
import mord
import numpy as np

boston = datasets.load_boston()
X, y = boston.data, np.round(boston.target).astype(np.int)
y -= y.min()

clf1 = linear_model.LogisticRegression(
    solver='lbfgs', multi_class='multinomial')
clf1.fit(X, y)
print('Mean Absolute Error of LogisticRegression: %s' %
      metrics.mean_absolute_error(clf1.predict(X), y))


clf2 = mord.LogisticAT(alpha=1.)
clf2.fit(X, y)
print('Mean Absolute Error of LogisticAT %s' %
      metrics.mean_absolute_error(clf2.predict(X), y))
