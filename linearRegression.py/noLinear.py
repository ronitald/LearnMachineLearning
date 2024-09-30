import matplotlib.pyplot as plt
import numpy as np

f, ax = plt.subplots()

x = np.linspace(0, 2*np.pi)
y = np.sin(x)
ax.plot(x, np.sin(x), 'r', label='sin ruido')
# añadimos algo de ruido
xr = x + np.random.normal(scale=0.1, size=x.shape)
yr = y + np.random.normal(scale=0.2, size=y.shape)
ax.scatter(xr, yr, label='con ruido')
ax.legend()

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

f, ax = plt.subplots()
ax.plot(x, np.sin(x), 'r', label='sin ruido')
ax.scatter(xr, yr, label='con ruido')

X = xr[:, np.newaxis]

for degree in [3, 4, 5]:
    model = make_pipeline(PolynomialFeatures(degree), Ridge())
    model.fit(X, y)
    y = model.predict(x[:, np.newaxis])
    ax.plot(x, y, '--', lw=2, label="degree %d" % degree)

ax.legend()

f, ax = plt.subplots()
ax.plot(x, np.sin(x), 'r', label='sin ruido')
ax.scatter(xr, yr, label='con ruido')

X = xr[:, np.newaxis]

for degree in [3, 4, 5]:
    model = make_pipeline(PolynomialFeatures(degree), RidgeCV(alphas=alphas))
    model.fit(X, y)
    y = model.predict(x[:, np.newaxis])
    ax.plot(x, y, '--', lw=2, label="degree %d" % degree)

ax.legend()