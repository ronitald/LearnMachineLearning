#Importar LogisticRegression
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(multi_class="multinomial")
clf.fit(x,y)
y_pred = clf.predict(x)