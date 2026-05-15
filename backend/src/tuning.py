from sklearn.model_selection import GridSearchCV # Es el optimizador del modelo (utiliza cross validation)
from sklearn.tree import DecisionTreeClassifier

def optimize_hyperparameters(X, y):
    param_grid = {
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'criterion': ['gini', 'entropy']
    }
    
    grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
    grid_search.fit(X, y)
    
    return grid_search.best_estimator_