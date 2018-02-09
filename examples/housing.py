import mord
from mord.datasets.base import load_housing
from sklearn import linear_model, metrics, preprocessing

data = load_housing()
features = data.data

le = preprocessing.LabelEncoder()
le.fit(data.target)
data.target = le.transform(data.target)

features.loc[features.Infl == 'Low', 'Infl'] = 1
features.loc[features.Infl == 'Medium', 'Infl'] = 2
features.loc[features.Infl == 'High', 'Infl'] = 3

features.loc[features.Cont == 'Low', 'Cont'] = 1
features.loc[features.Cont == 'Medium', 'Cont'] = 2
features.loc[features.Cont == 'High', 'Cont'] = 3

le = preprocessing.LabelEncoder()
le.fit(features.loc[:,'Type'])
features.loc[:,'type_encoded'] = le.transform(features.loc[:,'Type'])

X, y = features.loc[:,('Infl', 'Cont', 'type_encoded')], data.target

clf1 = linear_model.LogisticRegression(
    solver='lbfgs',
    multi_class='multinomial')
clf1.fit(X, y)

print('Mean Absolute Error of LogisticRegression: %s' %
      metrics.mean_absolute_error(clf1.predict(X), y))

clf2 = mord.LogisticAT(alpha=1.)
clf2.fit(X, y)
print('Mean Absolute Error of LogisticAT %s' %
      metrics.mean_absolute_error(clf2.predict(X), y))

clf3 = mord.LogisticIT(alpha=1.)
clf3.fit(X, y)
print('Mean Absolute Error of LogisticIT %s' %
      metrics.mean_absolute_error(clf3.predict(X), y))

clf4 = mord.LogisticSE(alpha=1.)
clf4.fit(X, y)
print('Mean Absolute Error of LogisticSE %s' %
      metrics.mean_absolute_error(clf4.predict(X), y))
