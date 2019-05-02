from functools import wraps
from time import time
from skopt import BayesSearchCV
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV

X, y = load_digits(10, True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.75, test_size=.25, random_state=0)

search_space = {
    'C': (1e-6, 1e+6, 'log-uniform'),
    'gamma': (1e-6, 1e+1, 'log-uniform'),
    'degree': (1, 8),  # integer valued parameter
    'kernel': ['linear', 'poly', 'rbf'],  # categorical parameter
}


def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        print('Elapsed time: {}'.format(end-start))
        return result
    return wrapper


# log-uniform: understand as search over p = exp(x) by varying x
opt = BayesSearchCV(
    SVC(),
    search_space,
    n_iter=32,
    cv=3,
    n_jobs=-1
)


@timing
def dummy(func, X_train, y_train):
    func(X_train, y_train)
    return None


dummy(opt.fit, X_train, y_train)

print("val. score: {}".format(opt.best_score_))
print("test score: {}".format(opt.score(X_test, y_test)))

param_grid = {
    'C': [1e-6, 1e-3, 1, 1e3, 1e+6],
    'gamma': [1e-6, 1e-4, 1e-2, 1.0, 1e+1],
    'degree': [1, 4, 8],  # integer valued parameter
    'kernel': ['linear', 'poly', 'rbf'],  # categorical parameter
}


grid_opt = GridSearchCV(
    SVC(),
    param_grid,
    cv=3,
    n_jobs=-1,
    return_train_score=True
)
dummy(grid_opt.fit, X_train, y_train)


print("val. score: {}".format(grid_opt.best_score_))
print("test score: {}".format(grid_opt.score(X_test, y_test)))
