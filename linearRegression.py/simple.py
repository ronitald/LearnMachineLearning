from sklearn import datasets
boston = datasets.load_boston()

print(boston.DESCR)

from sklearn.linear_model import LinearRegression
lr = LinearRegression(normalize=True)

lr.fit(boston.data, boston.target)

for (feature, coef) in zip(boston.feature_names, lr.coef_):
    print('{:>7}: {: 9.5f}'.format(feature, coef))

import matplotlib.pyplot as plt
import numpy as np

def plot_feature(feature):
    f = (boston.feature_names == feature)
    plt.scatter(boston.data[:,f], boston.target, c='b', alpha=0.3)
    plt.plot(boston.data[:,f], boston.data[:,f]*lr.coef_[f] + lr.intercept_, 'k')
    plt.legend(['Predicted value', 'Actual value'])
    plt.xlabel(feature)
    plt.ylabel("Median value in $1000's")

predictions = lr.predict(boston.data)
f, ax = plt.subplots(1)

ax.hist(boston.target - predictions, bins=50, alpha=0.7)
ax.set_title('Histograma de residuales')
ax.text(0.95, 0.90, 'Media de residuales: {:.3e}'.format(np.mean(boston.target - predictions)),
        transform=ax.transAxes, verticalalignment='top', horizontalalignment='right')

from sklearn.datasets import make_regression
reg_data, reg_target = make_regression(n_samples=2000, n_features=3, effective_rank=2, noise=10)

from sklearn.linear_model import RidgeCV

alphas = np.linspace(0.01, 0.5)

rcv = RidgeCV(alphas=alphas, store_cv_values=True)
rcv.fit(reg_data, reg_target)

plt.rc('text', usetex=False)
f, ax = plt.subplots()

ax.plot(alphas, rcv.cv_values_.mean(axis=0))
ax.text(0.05, 0.90, 'alpha que minimiza el error: {:.3f}'.format(rcv.alpha_),
        transform=ax.transAxes)