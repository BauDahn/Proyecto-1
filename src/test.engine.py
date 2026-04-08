from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as SklearnTree
from Tree.tree import DecisionTree as MyTree
import numpy as np

# 1. Carga de datos
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Nuestro árbol
my_model = MyTree(max_depth=5, min_samples=2, cantidad_minima_en_nodo=1)
my_model.fit(X_train, y_train)
my_preds = my_model.predict(X_test)
my_acc = np.mean(my_preds == y_test)

# 3. Árbol de Scikit-learn (Referencia)
sk_model = SklearnTree(max_depth=5, min_samples_split=2)
sk_model.fit(X_train, y_train)
sk_acc = sk_model.score(X_test, y_test)

print(f"Precisión de nuestro motor: {my_acc:.4f}")
print(f"Precisión de Scikit-learn: {sk_acc:.4f}")