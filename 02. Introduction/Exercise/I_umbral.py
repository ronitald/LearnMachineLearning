# Importar archivo Main
from H_accuracyRecall import *

# Puntación de un paciente aleatorio
y_score = sgd.decision_function([[47,1,1,0,8,0,0]])
y_score

# Graficar la precisión y recall
y_scores = cross_val_predict(sgd,x_train,y_train_y,cv=3,method="decision_function")

# Puedes visualizar los y_scores pero no te dice nada, la gráfica si lo hará
# Graficar la precisión y recall, ahora si
from sklearn.metrics import precision_recall_curve
precisions, recalls, umbrales = precision_recall_curve(y_train_y,y_scores)
plt.plot(umbrales, precisions[:-1],"b--",label="Precisión")
plt.plot(umbrales, recalls[:-1],"g-",label="Recall")
plt.legend()
plt.show()

# Supon que ya lo pensaste y querías un umbral  90
umbral_90 = umbrales[np.argmax(precisions >= 0.90)]
umbral_90

# Arroja la precisión y recall para un umbral de 90
y_train_90 = (y_scores >= umbral_90)
p = precision_score(y_train_y,y_train_90)
r = recall_score(y_train_y,y_train_90)
p,r