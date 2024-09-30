import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generar datos sintéticos para el problema de clasificación de spam
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Función para visualizar las curvas de aprendizaje
def plot_learning_curve(estimator, X, y):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring='accuracy'
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title("Curva de Aprendizaje")
    plt.xlabel("Número de ejemplos de entrenamiento")
    plt.ylabel("Precisión")
    plt.grid()

    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r"
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g"
    )

    plt.plot(
        train_sizes,
        train_scores_mean,
        'o-',
        color="r",
        label="Entrenamiento"
    )
    plt.plot(
        train_sizes,
        test_scores_mean,
        'o-',
        color="g",
        label="Validación"
    )

    plt.legend(loc="best")
    plt.show()

# Crear y entrenar el modelo de regresión logística
estimator = LogisticRegression()
plot_learning_curve(estimator, X, y)